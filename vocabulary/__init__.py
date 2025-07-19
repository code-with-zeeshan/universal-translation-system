# vocabulary/__init__.py
"""
Vocabulary management module for the Universal Translation System.

Handles dynamic vocabulary loading, optimization, and language-specific
vocabulary packs for efficient multilingual translation.
"""

from .vocabulary_manager import VocabularyManager, VocabularyPack
from .optimized_vocab_manager import OptimizedVocabularyManager, EdgeVocabularyPack
from .create_vocabulary_packs_from_data import VocabularyPackCreator

__all__ = [
    "VocabularyManager",
    "VocabularyPack",
    "OptimizedVocabularyManager",
    "EdgeVocabularyPack",
    "VocabularyPackCreator"
]