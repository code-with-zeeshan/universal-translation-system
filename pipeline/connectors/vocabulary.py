# pipeline/connectors/vocabulary.py
"""Connect data pipeline to vocabulary creation"""
from pathlib import Path
import logging
from typing import Optional

from utils.common_utils import RuntimeDirectoryManager
from utils.exceptions import DataError



class VocabularyConnector:
    """Bridge between data pipeline and vocabulary creation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.runtime_dirs = RuntimeDirectoryManager()
    
    def create_vocabularies_from_pipeline(self, processed_dir: str = '',
                                           output_dir: str = '',
                                           vocab_size: int = 32000,
                                           max_model_vocab_size: Optional[int] = None):
        """Create vocabulary packs after data pipeline completes"""
        if not output_dir:
            output_dir = str(self.runtime_dirs.vocab_dir)
        if not processed_dir:
            processed_dir = str(self.runtime_dirs.processed_dir)
        from pipeline.vocabulary.creator import UnifiedVocabularyCreator as VocabularyPackCreator
        from pipeline.vocabulary.config import UnifiedVocabConfig

        # Validate pack vocab_size won't exceed model embedding capacity
        if max_model_vocab_size is not None and vocab_size > max_model_vocab_size:
            raise DataError(
                f"Vocabulary vocab_size ({vocab_size}) exceeds model "
                f"max_vocab_size ({max_model_vocab_size}). "
                f"Tokens would produce out-of-range embedding indices."
            )
        
        self.logger.info(f"Creating vocabulary packs in {output_dir} from {processed_dir}...")
        
        # Check if monolingual corpora exist
        processed_path = Path(processed_dir)
        if not processed_path.exists():
            raise DataError(f"Processed data directory not found: {processed_dir}")
        
        # Create vocabulary packs with requested vocab size per pack
        creator = VocabularyPackCreator(
            corpus_dir=str(self.runtime_dirs.corpus_dir),
            output_dir=output_dir,
            config=UnifiedVocabConfig(vocab_size=vocab_size),
        )
        
        # Create all standard packs
        created_packs = creator.create_all_packs()
        
        self.logger.info(f"Created {len(created_packs)} vocabulary packs")
        return created_packs
