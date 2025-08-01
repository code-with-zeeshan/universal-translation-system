# utils/interfaces.py
"""
Interfaces to break circular dependencies
Uses Protocol classes for type hints without imports
"""
from typing import Protocol, Dict, List, Any, Optional

class VocabularyInterface(Protocol):
    """Interface for vocabulary operations"""
    
    def create_vocabularies_from_pipeline(self, processed_dir: str) -> List[Dict[str, Any]]:
        """Create vocabulary packs from processed data"""
        ...
    
    def get_vocab_for_pair(self, source_lang: str, target_lang: str) -> Any:
        """Get vocabulary for language pair"""
        ...

class DataProcessorInterface(Protocol):
    """Interface for data processing operations"""
    
    def process_streaming_dataset(self, dataset: Any, output_path: Any, 
                                 batch_size: int = 1000,
                                 max_samples: Optional[int] = None) -> int:
        """Process streaming dataset"""
        ...