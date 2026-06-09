# connector/vocabulary_connector.py
"""Connect data pipeline to vocabulary creation"""
from pathlib import Path
import logging

from utils.exceptions import DataError
from utils.constants import VOCAB_DIR, DATA_CORPUS_DIR


class VocabularyConnector:
    """Bridge between data pipeline and vocabulary creation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_vocabularies_from_pipeline(self, processed_dir: str = 'data/processed',
                                           output_dir: str = '',
                                           vocab_size: int = 32000):
        """Create vocabulary packs after data pipeline completes"""
        if not output_dir:
            output_dir = VOCAB_DIR
        from vocabulary.unified_vocabulary_creator import UnifiedVocabularyCreator as VocabularyPackCreator
        from vocabulary.vocab_config import UnifiedVocabConfig
        
        self.logger.info(f"Creating vocabulary packs in {output_dir} from {processed_dir}...")
        
        # Check if monolingual corpora exist
        processed_path = Path(processed_dir)
        if not processed_path.exists():
            raise DataError(f"Processed data directory not found: {processed_dir}")
        
        # Create vocabulary packs with requested vocab size per pack
        creator = VocabularyPackCreator(
            corpus_dir=str(Path(processed_dir) / DATA_CORPUS_DIR),
            output_dir=output_dir,
            config=UnifiedVocabConfig(vocab_size=vocab_size),
        )
        
        # Create all standard packs
        created_packs = creator.create_all_packs()
        
        self.logger.info(f"Created {len(created_packs)} vocabulary packs")
        return created_packs