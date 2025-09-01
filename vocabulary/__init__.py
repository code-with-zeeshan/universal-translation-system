# vocabulary/__init__.py
"""
Vocabulary management module for the Universal Translation System.

Handles dynamic vocabulary loading, optimization, and language-specific
vocabulary packs for efficient multilingual translation.
"""

from .unified_vocab_manager import UnifiedVocabularyManager, VocabularyPack, VocabularyMode
from .unified_vocabulary_creator import UnifiedVocabularyCreator as VocabularyPackCreator

# Backward compatibility aliases
VocabularyManager = UnifiedVocabularyManager
OptimizedVocabularyManager = UnifiedVocabularyManager
EdgeVocabularyPack = VocabularyPack

__all__ = [
    "VocabularyManager",
    "VocabularyPack",
    "OptimizedVocabularyManager",
    "EdgeVocabularyPack",
    "VocabularyPackCreator"
]

from vocabulary.unified_vocab_manager import UnifiedVocabularyManager, VocabularyMode

def create_vocabulary_manager(config, context=None):
    """Factory function to create appropriate vocabulary manager."""
    
    # Determine mode based on config or context
    if hasattr(config, 'deployment_mode'):
        mode_map = {
            'edge': VocabularyMode.EDGE,
            'mobile': VocabularyMode.OPTIMIZED,
            'server': VocabularyMode.FULL
        }
        mode = mode_map.get(config.deployment_mode, VocabularyMode.FULL)
    else:
        # Default based on available resources
        import torch
        if not torch.cuda.is_available():
            mode = VocabularyMode.EDGE  # CPU only = use edge mode
        elif torch.cuda.get_device_properties(0).total_memory < 8e9:
            mode = VocabularyMode.OPTIMIZED  # <8GB GPU
        else:
            mode = VocabularyMode.FULL  # High-end GPU
    
    return UnifiedVocabularyManager(config, mode=mode)

# Example usage (do not execute at import time):
# from vocabulary import create_vocabulary_manager
# vocab_mgr = create_vocabulary_manager(your_config)