# connector/vocabulary_connector.py
"""Connect data pipeline to vocabulary creation"""
from pathlib import Path
import logging

from utils.exceptions import DataError


class VocabularyConnector:
    """Bridge between data pipeline and vocabulary creation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_vocabularies_from_pipeline(self, processed_dir: str = 'data/processed',
                                           output_dir: str = 'vocabs'):
        """Create vocabulary packs after data pipeline completes"""
        from vocabulary.unified_vocabulary_creator import UnifiedVocabularyCreator as VocabularyPackCreator
        
        self.logger.info(f"Creating vocabulary packs in {output_dir} from {processed_dir}...")
        
        # Check if monolingual corpora exist
        processed_path = Path(processed_dir)
        if not processed_path.exists():
            raise DataError(f"Processed data directory not found: {processed_dir}")
        
        # Create vocabulary packs
        creator = VocabularyPackCreator(
            corpus_dir=processed_dir,
            output_dir=output_dir,
        )
        
        # Create all standard packs
        created_packs = creator.create_all_packs()
        
        self.logger.info(f"Created {len(created_packs)} vocabulary packs")
        return created_packs