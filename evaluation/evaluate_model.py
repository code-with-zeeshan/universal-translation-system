"""
Comprehensive evaluation module for the Universal Translation System
Supports BLEU, COMET, and custom metrics
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch

from evaluation.metrics import TranslationPair
from evaluation.evaluator import TranslationEvaluator, evaluate_translation_quality
from config.schemas import load_config as load_pydantic_config
from utils.constants import (
    LOG_DIR, MODELS_ENCODER_DIR, MODELS_DECODER_DIR,
    VOCAB_DIR, EVALUATION_REPORT_FILENAME
)
from utils.logging_config import setup_logging

setup_logging(log_dir=LOG_DIR, log_level="INFO")
logger = logging.getLogger(__name__)


def build_encoder(config) -> torch.nn.Module:
    """Build encoder from config."""
    from encoder.universal_encoder import UniversalEncoder
    return UniversalEncoder(
        max_vocab_size=config.model.vocab_size,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
    )


def build_decoder(config) -> torch.nn.Module:
    """Build decoder from config."""
    from cloud_decoder import OptimizedUniversalDecoder
    return OptimizedUniversalDecoder(
        encoder_dim=config.model.hidden_dim,
        decoder_dim=config.model.decoder_dim,
        vocab_size=config.model.vocab_size,
        num_layers=config.model.decoder_layers,
        num_heads=config.model.decoder_heads,
        dropout=config.model.dropout,
        max_length=config.model.max_seq_length,
    )


def wrap_with_lora(encoder, decoder, config) -> tuple:
    """Wrap models with LoRA if configured. Returns (encoder, decoder)."""
    if not getattr(config.training, 'use_lora', False):
        return encoder, decoder

    from training.peft_integration import wrap_encoder_with_lora, wrap_decoder_with_lora

    encoder = wrap_encoder_with_lora(
        encoder,
        r=config.training.lora_r,
        lora_alpha=config.training.lora_alpha,
        lora_dropout=config.training.lora_dropout,
        use_rslora=getattr(config.training, 'use_rslora', True),
    )

    decoder = wrap_decoder_with_lora(
        decoder,
        r=config.training.lora_r_decoder,
        lora_alpha=config.training.lora_alpha,
        lora_dropout=config.training.lora_dropout,
        use_rslora=getattr(config.training, 'use_rslora', True),
    )

    return encoder, decoder


def load_checkpoint(
    checkpoint_path: str,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """Load model weights from checkpoint.

    Returns the full checkpoint dict for metadata inspection.
    """
    logger.info(f"📥 Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Determine if this is a full training checkpoint or a standalone model
    if 'encoder_state_dict' in ckpt and 'decoder_state_dict' in ckpt:
        enc_sd = ckpt['encoder_state_dict']
        dec_sd = ckpt['decoder_state_dict']
        # Strip torch.compile prefix _orig_mod. but KEEP PEFT prefix base_model.model.
        # The checkpoint was saved from a compiled+PEFT model, giving keys like:
        #   _orig_mod.base_model.model.transformer_layers.0.q_proj.lora_A.default.weight
        # The PEFT-wrapped model expects: base_model.model.transformer_layers.0.q_proj.lora_A.default.weight
        enc_sd = {k.removeprefix('_orig_mod.'): v for k, v in enc_sd.items()}
        dec_sd = {k.removeprefix('_orig_mod.'): v for k, v in dec_sd.items()}
        enc_missing, enc_unexpected = encoder.load_state_dict(enc_sd, strict=False)
        dec_missing, dec_unexpected = decoder.load_state_dict(dec_sd, strict=False)
        if enc_missing:
            logger.warning(f"Encoder missing keys: {enc_missing}")
        if enc_unexpected:
            logger.warning(f"Encoder unexpected keys: {enc_unexpected}")
        if dec_missing:
            logger.warning(f"Decoder missing keys: {dec_missing}")
        if dec_unexpected:
            logger.warning(f"Decoder unexpected keys: {dec_unexpected}")
        val_loss = ckpt.get('best_val_loss', 'N/A')
        logger.info(f"✅ Checkpoint loaded (best_val_loss={val_loss})")
    elif 'model_state_dict' in ckpt:
        encoder.load_state_dict(ckpt['model_state_dict'], strict=False)
        decoder.load_state_dict(ckpt['model_state_dict'], strict=False)
        logger.info("✅ Single model_state_dict checkpoint loaded")
    else:
        # Try loading as full model state dict
        try:
            encoder.load_state_dict(ckpt, strict=False)
            decoder.load_state_dict(ckpt, strict=False)
            logger.info("✅ Checkpoint loaded directly as state dict")
        except Exception as e:
            logger.warning(f"Could not load checkpoint directly: {e}")

    return ckpt


def find_test_files(test_data_dir: str) -> List[Path]:
    """Scan test_data directory for evaluation files."""
    test_path = Path(test_data_dir)
    if not test_path.exists():
        logger.error(f"Test data directory not found: {test_data_dir}")
        return []

    files = sorted(test_path.glob('*.tsv')) + sorted(test_path.glob('*.txt'))
    if not files:
        logger.warning(f"No .tsv or .txt files found in {test_data_dir}")
    else:
        logger.info(f"Found {len(files)} test file(s) in {test_data_dir}")
        for f in files:
            logger.info(f"  - {f.name}")

    return files


def evaluate_file(
    evaluator: TranslationEvaluator,
    test_file: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run evaluation on a single test file and save per-file report."""
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 Evaluating {test_file.name}...")
    logger.info(f"{'='*60}")

    try:
        metrics = evaluator.evaluate_file(str(test_file), file_format='tsv')

        # Save per-file report
        report = evaluator.create_evaluation_report(
            metrics,
            str(output_dir / f"report_{test_file.stem}.json")
        )
        report['_file'] = test_file.name

        lang_pair_scores = metrics.get('language_pair_scores', {})
        if lang_pair_scores:
            logger.info("Per-language-pair BLEU:")
            for pair, score in sorted(lang_pair_scores.items()):
                logger.info(f"  {pair}: BLEU={score.get('bleu', 'N/A')} "
                          f"({score.get('sample_count', 0)} samples)")

        logger.info(f"✅ {test_file.name} evaluation complete")
        return report

    except Exception as e:
        logger.error(f"❌ Failed to evaluate {test_file.name}: {e}")
        return {'error': str(e), 'file': test_file.name}


def main(
    config: Optional[str] = None,
    checkpoint: Optional[str] = None,
    test_data: Optional[str] = None,
) -> bool:
    """Run full evaluation pipeline.

    Args:
        config: Path to config YAML
        checkpoint: Path to model checkpoint (.pt file)
        test_data: Path to test data directory or file

    Returns:
        True on success, False on failure
    """
    # Default paths
    config = config or 'config/base.yaml'
    checkpoint = checkpoint or 'models/production/best_model.pt'
    test_data = test_data or 'data/evaluation'

    # Load config
    logger.info(f"📄 Loading config from {config}")
    try:
        cfg = load_pydantic_config(config)
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        return False

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"💻 Device: {device}")

    # Build models
    logger.info("🔧 Building models...")
    encoder = build_encoder(cfg)
    decoder = build_decoder(cfg)

    # Add target language adapters (must be done before LoRA wrapping)
    if hasattr(cfg.data, 'languages'):
        for lang in cfg.data.languages:
            if lang != 'en':
                decoder.add_target_language_adapter(lang)

    # Wrap with LoRA if configured
    encoder, decoder = wrap_with_lora(encoder, decoder, cfg)

    # Move to device
    encoder.to(device)
    decoder.to(device)

    # Load checkpoint
    if not Path(checkpoint).exists():
        logger.error(f"❌ Checkpoint not found: {checkpoint}")
        return False

    ckpt = load_checkpoint(checkpoint, encoder, decoder, device)

    # Load vocabulary manager
    logger.info("📚 Loading vocabulary...")
    try:
        from vocabulary.unified_vocab_manager import UnifiedVocabularyManager, VocabularyMode
        vocab_dir = cfg.vocabulary.vocab_dir
        vocab_path = Path(vocab_dir)
        if not vocab_path.exists() or not list(vocab_path.glob('*_v*.msgpack')):
            logger.error(f"❌ Vocabulary packs not found at {vocab_dir}")
            logger.error("   Run vocabulary creation first:")
            logger.error(f"     python -m data.pipeline_orchestrator --stage vocabulary")
            logger.error("   Or transfer existing vocab packs:")
            logger.error(f"     cp /path/to/vocab/*.msgpack {vocab_dir}/")
            return False

        vocab_manager = UnifiedVocabularyManager(
            cfg,
            vocab_dir=vocab_dir,
            mode=VocabularyMode.FULL,
        )
        if vocab_manager is None:
            logger.error("❌ Failed to create vocabulary manager")
            return False
    except Exception as e:
        logger.error(f"❌ Vocabulary loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Create evaluator
    logger.info("🔬 Initializing evaluator...")
    evaluator = TranslationEvaluator(
        encoder_model=encoder,
        decoder_model=decoder,
        vocabulary_manager=vocab_manager,
        device=device,
    )

    # Find test files
    test_path = Path(test_data)
    if test_path.is_file():
        test_files = [test_path]
    else:
        test_files = find_test_files(test_data)

    if not test_files:
        logger.error("❌ No test files found. Run `python -m data.pipeline_orchestrator --eval-only` to download evaluation data.")
        return False

    # Create output directory for reports
    output_dir = Path('evaluation_reports')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate each file
    all_reports = []
    for test_file in test_files:
        report = evaluate_file(evaluator, test_file, output_dir)
        all_reports.append(report)

    # Generate combined report
    combined = {
        'config': config,
        'checkpoint': checkpoint,
        'test_data': str(test_data),
        'num_files': len(test_files),
        'files_evaluated': [str(f) for f in test_files],
        'reports': all_reports,
        'best_val_loss': float(ckpt.get('best_val_loss', 0)),
        'global_step': ckpt.get('global_step', 0),
        'epoch': ckpt.get('epoch', 0),
    }

    combined_path = output_dir / EVALUATION_REPORT_FILENAME
    with open(combined_path, 'w') as f:
        json.dump(combined, f, indent=2, default=str)
    logger.info(f"📄 Combined report saved to {combined_path}")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("📊 OVERALL EVALUATION SUMMARY")
    logger.info("="*60)
    for report in all_reports:
        if 'summary' in report:
            file_name = report.get('_file', 'unknown')
            logger.info(f"\n  {file_name}:")
            for metric, value in report['summary'].items():
                logger.info(f"    {metric}: {value:.4f}")
    logger.info("="*60)

    return True


__all__ = [
    "TranslationEvaluator",
    "TranslationPair",
    "evaluate_translation_quality",
]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument('--config', default='config/base.yaml')
    parser.add_argument('--checkpoint', default='models/production/best_model.pt')
    parser.add_argument('--test-data', default='data/evaluation')
    args = parser.parse_args()
    success = main(
        config=args.config,
        checkpoint=args.checkpoint,
        test_data=args.test_data,
    )
    sys.exit(0 if success else 1)
