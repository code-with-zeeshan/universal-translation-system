#!/usr/bin/env python3
"""
Unified CLI for Universal Translation System pipelines

Supports modes:
- data: Run unified data pipeline (select stages or all)
- vocab: Create vocabulary packs using unified creator
- bootstrap: Bootstrap encoder/decoder from pretrained models
- train: Train via training.launch (internal call)
- evaluate: Evaluate via training.launch (internal call)
- profile: Profile via training.launch (internal call)
- compare: Compare experiments via training.launch (internal call)
- convert: Convert models (PyTorch->ONNX, ONNX->CoreML/TFLite, etc.)
- all: Run data -> vocab -> bootstrap -> train -> convert sequentially

Usage examples:
- python scripts/pipeline.py data --config config/training_generic_gpu.yaml
- python scripts/pipeline.py vocab --mode production --groups latin cjk
- python scripts/pipeline.py train --config config/training_generic_gpu.yaml --distributed
- python scripts/pipeline.py all --config config/training_generic_gpu.yaml
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import List, Optional
import logging

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Common config
from config.schemas import load_config as load_pydantic_config, RootConfig

# Data pipeline
from data.unified_data_pipeline import UnifiedDataPipeline, PipelineStage

# Vocabulary
from vocabulary.unified_vocabulary_creator import (
    UnifiedVocabularyCreator,
    UnifiedVocabConfig,
    CreationMode,
)

# Training launcher (for train/evaluate/profile/compare)
from training import launch as training_launch

# Bootstrap & Conversion
from training.bootstrap_from_pretrained import PretrainedModelBootstrapper
from training.convert_models import ModelConverter

logger = logging.getLogger("pipeline")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ====================== COMMAND IMPLEMENTATIONS ======================

def run_data_pipeline(config_path: str, stages: Optional[List[str]] = None, resume: bool = True) -> None:
    """Run the unified data pipeline with optional stage selection."""
    cfg: RootConfig = load_pydantic_config(config_path)
    pipeline = UnifiedDataPipeline(cfg)

    # Map stage names to PipelineStage enum
    selected_stages = None
    if stages:
        name_to_stage = {s.value: s for s in PipelineStage}
        try:
            selected_stages = [name_to_stage[s] for s in stages]
        except KeyError as e:
            valid = ", ".join(sorted(name_to_stage.keys()))
            raise SystemExit(f"Invalid stage '{e.args[0]}'. Valid stages: {valid}")

    asyncio.run(pipeline.run_pipeline(resume=resume, stages=selected_stages))


def run_vocab_creator(
    corpus_dir: str,
    output_dir: str,
    mode: str = "production",
    groups: Optional[List[str]] = None,
    vocab_size: Optional[int] = None,
    model_type: Optional[str] = None,
    character_coverage: Optional[float] = None,
    num_threads: Optional[int] = None,
) -> None:
    """Run the unified vocabulary creator for specified groups or all."""
    # Build config
    config = UnifiedVocabConfig()
    if vocab_size is not None:
        config.vocab_size = vocab_size
    if model_type is not None:
        config.model_type = model_type
    if character_coverage is not None:
        config.character_coverage = character_coverage
    if num_threads is not None:
        config.num_threads = num_threads

    # Mode
    try:
        creation_mode = CreationMode(mode.lower())
    except ValueError:
        valid = ", ".join([m.value for m in CreationMode])
        raise SystemExit(f"Invalid mode '{mode}'. Valid: {valid}")

    creator = UnifiedVocabularyCreator(
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        config=config,
        default_mode=creation_mode,
    )

    creator.create_all_packs(mode=creation_mode, groups_to_create=groups)


def run_bootstrap(
    encoder_model: str = 'xlm-roberta-base',
    decoder_model: str = 'facebook/mbart-large-50',
    encoder_out: str = 'models/encoder/universal_encoder_initial.pt',
    decoder_out: str = 'models/decoder/universal_decoder_initial.pt',
    target_hidden_dim: int = 1024,
    device: str = 'auto',
) -> None:
    """Bootstrap encoder and decoder from pretrained sources."""
    bs = PretrainedModelBootstrapper(device=device)
    bs.create_encoder_from_pretrained(
        model_name=encoder_model,
        output_path=encoder_out,
        target_hidden_dim=target_hidden_dim,
    )
    bs.create_decoder_from_mbart(
        model_name=decoder_model,
        output_path=decoder_out,
    )


def run_training(
    config_path: str,
    experiment_name: Optional[str] = None,
    checkpoint: Optional[str] = None,
    distributed: bool = False,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    num_epochs: Optional[int] = None,
    log_dir: str = 'logs',
    log_level: str = 'info',
) -> None:
    """Delegate to training.launch.launch_training with a constructed namespace."""
    ns = argparse.Namespace(
        config=config_path,
        experiment_name=experiment_name,
        checkpoint=checkpoint,
        distributed=distributed,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        log_dir=log_dir,
        log_level=log_level,
    )
    training_launch.launch_training(ns)


def run_evaluation(config_path: str, checkpoint: str, test_data: Optional[str], batch_size: Optional[int], output_dir: str) -> None:
    ns = argparse.Namespace(
        config=config_path,
        checkpoint=checkpoint,
        test_data=test_data,
        batch_size=batch_size,
        output_dir=output_dir,
    )
    training_launch.launch_evaluation(ns)


def run_profile(config_path: str, profile_steps: int, benchmark: bool, output_dir: str) -> None:
    ns = argparse.Namespace(
        config=config_path,
        profile_steps=profile_steps,
        benchmark=benchmark,
        output_dir=output_dir,
    )
    training_launch.launch_profiling(ns)


def run_compare(experiments: List[str], output_dir: str) -> None:
    comparator = training_launch
    # Use the launcher subcommand pathway for consistency
    from training.comparison import ExperimentComparator
    comp = ExperimentComparator(experiments, output_dir)
    comp.generate_comparison_report()
    comp.plot_learning_curves()
    comp.plot_metrics_comparison()


def run_convert(
    task: str,
    model_path: Optional[str] = None,
    onnx_path: Optional[str] = None,
    output_path: Optional[str] = None,
    opset: int = 17,
    use_dynamo: bool = True,
    tflite_quantize: bool = True,
    coreml_min_target: str = "15.0",
) -> None:
    """Model conversion utilities wrapping ModelConverter."""
    conv = ModelConverter()

    if task == 'pytorch-to-onnx':
        if not (model_path and output_path):
            raise SystemExit("pytorch-to-onnx requires --model-path and --output-path")
        import torch
        dummy = torch.randn(1, 512)
        ok = conv.pytorch_to_onnx(model_path, output_path, dummy, opset_version=opset, use_dynamo=use_dynamo)
        if not ok:
            raise SystemExit(1)
    elif task == 'onnx-to-coreml':
        if not (onnx_path and output_path):
            raise SystemExit("onnx-to-coreml requires --onnx-path and --output-path")
        ok = conv.onnx_to_coreml(onnx_path, output_path, minimum_deployment_target=coreml_min_target)
        if not ok:
            raise SystemExit(1)
    elif task == 'onnx-to-tflite':
        if not (onnx_path and output_path):
            raise SystemExit("onnx-to-tflite requires --onnx-path and --output-path")
        ok = conv.onnx_to_tflite(onnx_path, output_path, quantize=tflite_quantize)
        if not ok:
            raise SystemExit(1)
    else:
        raise SystemExit("Invalid convert task. Use one of: pytorch-to-onnx, onnx-to-coreml, onnx-to-tflite")


def run_all(config_path: str) -> None:
    """Run full pipeline: data -> vocab -> bootstrap -> train -> convert (ONNX)."""
    # 1) Data
    logger.info("[ALL] Running data pipeline...")
    run_data_pipeline(config_path, stages=None, resume=True)

    # 2) Vocabulary (defaults: all groups, production)
    logger.info("[ALL] Creating vocabulary packs...")
    cfg = load_pydantic_config(config_path)
    run_vocab_creator(
        corpus_dir=cfg.data.processed_dir,
        output_dir=cfg.vocabulary.vocab_dir,
        mode="production",
        groups=None,
    )

    # 3) Bootstrap
    logger.info("[ALL] Bootstrapping pretrained models...")
    run_bootstrap()

    # 4) Train
    logger.info("[ALL] Training...")
    run_training(config_path=config_path)

    # 5) Convert to ONNX (encoder example)
    logger.info("[ALL] Converting encoder to ONNX...")
    run_convert(
        task='pytorch-to-onnx',
        model_path='models/encoder/universal_encoder_initial.pt',
        output_path='models/encoder/universal_encoder.onnx',
        opset=17,
        use_dynamo=True,
    )


# ====================== ARGPARSE ======================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified pipeline CLI for Universal Translation System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest='command', required=True)

    # data
    sp = sub.add_parser('data', help='Run unified data pipeline')
    sp.add_argument('--config', type=str, required=True, help='Path to config YAML')
    sp.add_argument('--stages', nargs='*', help='Subset of stages to run (names). Omit to run all.')
    sp.add_argument('--no-resume', action='store_true', help='Do not resume from checkpoint')

    # vocab
    sp = sub.add_parser('vocab', help='Create vocabulary packs')
    sp.add_argument('--corpus-dir', type=str, default='data/processed', help='Input corpus directory')
    sp.add_argument('--output-dir', type=str, default='vocabs', help='Output directory for packs')
    sp.add_argument('--mode', type=str, default='production', choices=[m.value for m in CreationMode], help='Creation mode')
    sp.add_argument('--groups', nargs='*', help='Specific groups to create (default: all)')
    sp.add_argument('--vocab-size', type=int, help='Override vocab size')
    sp.add_argument('--model-type', type=str, help='SentencePiece model type (bpe/unigram/...)')
    sp.add_argument('--character-coverage', type=float, help='SentencePiece character coverage')
    sp.add_argument('--num-threads', type=int, help='SPM threads')

    # bootstrap
    sp = sub.add_parser('bootstrap', help='Bootstrap encoder/decoder from pretrained models')
    sp.add_argument('--encoder-model', type=str, default='xlm-roberta-base')
    sp.add_argument('--decoder-model', type=str, default='facebook/mbart-large-50')
    sp.add_argument('--encoder-out', type=str, default='models/encoder/universal_encoder_initial.pt')
    sp.add_argument('--decoder-out', type=str, default='models/decoder/universal_decoder_initial.pt')
    sp.add_argument('--target-hidden-dim', type=int, default=1024)
    sp.add_argument('--device', type=str, default='auto')

    # train
    sp = sub.add_parser('train', help='Train model (delegates to training.launch)')
    sp.add_argument('--config', type=str, required=True)
    sp.add_argument('--experiment-name', type=str)
    sp.add_argument('--checkpoint', type=str)
    sp.add_argument('--distributed', action='store_true')
    sp.add_argument('--batch-size', type=int)
    sp.add_argument('--learning-rate', type=float)
    sp.add_argument('--num-epochs', type=int)
    sp.add_argument('--log-dir', type=str, default='logs')
    sp.add_argument('--log-level', type=str, default='info', choices=['debug','info','warning','error'])

    # evaluate
    sp = sub.add_parser('evaluate', help='Evaluate model (delegates to training.launch)')
    sp.add_argument('--config', type=str, required=True)
    sp.add_argument('--checkpoint', type=str, required=True)
    sp.add_argument('--test-data', type=str)
    sp.add_argument('--batch-size', type=int)
    sp.add_argument('--output-dir', type=str, default='results')

    # profile
    sp = sub.add_parser('profile', help='Profile training (delegates to training.launch)')
    sp.add_argument('--config', type=str, required=True)
    sp.add_argument('--profile-steps', type=int, default=10)
    sp.add_argument('--benchmark', action='store_true')
    sp.add_argument('--output-dir', type=str, default='profiling')

    # compare
    sp = sub.add_parser('compare', help='Compare experiments')
    sp.add_argument('--experiments', nargs='+', required=True)
    sp.add_argument('--output-dir', type=str, default='comparisons')

    # convert
    sp = sub.add_parser('convert', help='Model conversion utilities')
    sp.add_argument('--task', type=str, required=True, choices=['pytorch-to-onnx','onnx-to-coreml','onnx-to-tflite'])
    sp.add_argument('--model-path', type=str)
    sp.add_argument('--onnx-path', type=str)
    sp.add_argument('--output-path', type=str)
    sp.add_argument('--opset', type=int, default=17)
    sp.add_argument('--use-dynamo', action='store_true')
    sp.add_argument('--no-use-dynamo', dest='use_dynamo', action='store_false')
    sp.set_defaults(use_dynamo=True)
    sp.add_argument('--tflite-no-quant', dest='tflite_quantize', action='store_false')
    sp.set_defaults(tflite_quantize=True)
    sp.add_argument('--coreml-min-target', type=str, default='15.0')

    # all
    sp = sub.add_parser('all', help='Run full pipeline: data -> vocab -> bootstrap -> train -> convert')
    sp.add_argument('--config', type=str, required=True)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == 'data':
        run_data_pipeline(
            config_path=args.config,
            stages=args.stages,
            resume=(not args.no_resume),
        )
    elif args.command == 'vocab':
        run_vocab_creator(
            corpus_dir=args.corpus_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            groups=args.groups,
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            character_coverage=args.character_coverage,
            num_threads=args.num_threads,
        )
    elif args.command == 'bootstrap':
        run_bootstrap(
            encoder_model=args.encoder_model,
            decoder_model=args.decoder_model,
            encoder_out=args.encoder_out,
            decoder_out=args.decoder_out,
            target_hidden_dim=args.target_hidden_dim,
            device=args.device,
        )
    elif args.command == 'train':
        run_training(
            config_path=args.config,
            experiment_name=args.experiment_name,
            checkpoint=args.checkpoint,
            distributed=args.distributed,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            log_dir=args.log_dir,
            log_level=args.log_level,
        )
    elif args.command == 'evaluate':
        run_evaluation(
            config_path=args.config,
            checkpoint=args.checkpoint,
            test_data=args.test_data,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        )
    elif args.command == 'profile':
        run_profile(
            config_path=args.config,
            profile_steps=args.profile_steps,
            benchmark=args.benchmark,
            output_dir=args.output_dir,
        )
    elif args.command == 'compare':
        run_compare(
            experiments=args.experiments,
            output_dir=args.output_dir,
        )
    elif args.command == 'convert':
        run_convert(
            task=args.task,
            model_path=args.model_path,
            onnx_path=args.onnx_path,
            output_path=args.output_path,
            opset=args.opset,
            use_dynamo=args.use_dynamo,
            tflite_quantize=args.tflite_quantize,
            coreml_min_target=args.coreml_min_target,
        )
    elif args.command == 'all':
        run_all(config_path=args.config)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()