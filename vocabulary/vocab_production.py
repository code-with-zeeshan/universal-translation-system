import logging
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

    # Extract embeddings if available
    if hasattr(self, 'extract_embeddings_for_tokens'):
        embeddings = self.extract_embeddings_for_tokens(vocab_data['tokens'])
        vocab_data['embeddings'] = embeddings

    # Cleanup
    self._cleanup_temp_files(merged_corpus, model_path)

    return vocab_data


def _train_sentencepiece_model(self, corpus_file: str, pack_name: str) -> str:
    """Train SentencePiece model"""
    model_prefix = str(self.output_dir / f"temp_{pack_name}")

    train_args = [
        f'--input={corpus_file}',
        f'--model_prefix={model_prefix}',
        f'--vocab_size={self.config.vocab_size}',
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

    logger.info("Training SentencePiece model...")
    spm.SentencePieceTrainer.train(' '.join(train_args))

    model_path = f"{model_prefix}.model"
    if not Path(model_path).exists():
        raise VocabularyError("Model file not created")

    return model_path
