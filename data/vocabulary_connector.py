# data/vocabulary_connector.py
"""Connect data pipeline to vocabulary creation"""
from pathlib import Path
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vocabulary.create_vocabulary_packs_from_data import VocabularyPackCreator
else:
    VocabularyPackCreator = None

class VocabularyConnector:
    """Bridge between data pipeline and vocabulary creation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_vocabularies_from_pipeline(self, processed_dir: str = 'data/processed'):
        """Create vocabulary packs after data pipeline completes"""
        # Lazy import to avoid circular dependency
        from vocabulary.create_vocabulary_packs_from_data import VocabularyPackCreator
        self.logger.info("Creating vocabulary packs from processed data...")
        
        # Check if monolingual corpora exist
        processed_path = Path(processed_dir)
        if not processed_path.exists():
            raise DataError(f"Processed data directory not found: {processed_dir}")
        
        # Create vocabulary packs
        creator = VocabularyPackCreator(
            corpus_dir=processed_dir,
            output_dir='vocabs'
        )
        
        # Create all standard packs
        created_packs = creator.create_all_packs()
        
        self.logger.info(f"Created {len(created_packs)} vocabulary packs")
        return created_packs