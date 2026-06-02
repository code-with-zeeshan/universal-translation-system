import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.exceptions import VocabularyError

logger = logging.getLogger("vocabulary")

spm = None


def _create_sentencepiece_vocab(
    self,
    languages: List[str],
    pack_name: str,
    domain: Optional[str] = None
) -> Dict[str, Any]:
    """Create vocabulary using SentencePiece (production mode)"""
    logger.info("Using SentencePiece for vocabulary creation...")

    # Merge corpora
    merged_corpus = self._merge_corpora(languages, pack_name, domain)

    # Train SentencePiece model
    model_path = self._train_sentencepiece_model(merged_corpus, pack_name)

    # Create vocabulary mappings
    vocab_data = self._create_vocabulary_mappings(model_path, languages)

    # Save model permanently
    model_path = self._model_path
    permanent_model = str(Path(model_path).parent / f"{pack_name}.model")
    if Path(model_path).exists() and not Path(permanent_model).exists():
        import shutil
        shutil.copy2(model_path, permanent_model)
        logger.info(f"Saved SentencePiece model to {permanent_model}")
    # Clean up temp model + vocab files
    for p in [model_path, model_path.replace('.model', '.vocab')]:
        Path(p).unlink(missing_ok=True)

    # Cleanup temp corpus only
    self._cleanup_temp_files(merged_corpus)

    return vocab_data


def _train_sentencepiece_model(self, corpus_file: str, pack_name: str) -> str:
    """Train SentencePiece model"""
    model_prefix = str(self.output_dir / f"temp_{pack_name}")

    vocab_size = self.config.vocab_size

    train_args = [
        f'--input={corpus_file}',
        f'--model_prefix={model_prefix}',
        f'--vocab_size={vocab_size}',
        f'--model_type={self.config.model_type}',
        f'--character_coverage={self.config.character_coverage}',
        f'--pad_id={self.config.pad_id}',
        f'--unk_id={self.config.unk_id}',
        f'--bos_id={self.config.bos_id}',
        f'--eos_id={self.config.eos_id}',
        f'--num_threads={self.config.num_threads}',
        f'--train_extremely_large_corpus=true',
        f'--byte_fallback=true'
    ]

    # Lazy import to avoid requiring sentencepiece when not used
    global spm
    if spm is None:
        try:
            import sentencepiece as spm  # type: ignore
        except Exception as e:
            raise VocabularyError("sentencepiece not installed; required for PRODUCTION mode. Install with: pip install sentencepiece") from e

    logger.info(f"Training SentencePiece model with vocab_size={vocab_size}...")
    try:
        spm.SentencePieceTrainer.train(' '.join(train_args))
    except RuntimeError as e:
        # Auto-reduce vocab_size if it exceeds unique tokens in corpus
        match = re.search(r'value <= (\d+)', str(e))
        if match and vocab_size > 1000:
            max_allowed = int(match.group(1))
            reduced = min(max_allowed, max(1000, vocab_size // 2))
            logger.warning(f"vocab_size {vocab_size} too large for this corpus (max {max_allowed}), retrying with {reduced}")
            self.config.vocab_size = reduced
            return self._train_sentencepiece_model(corpus_file, pack_name)
        raise

    model_path = f"{model_prefix}.model"
    self._model_path = model_path
    if not Path(model_path).exists():
        raise VocabularyError("Model file not created")

    return model_path
