# vocabulary/unified_vocabulary_creator.py
"""
Re-export shim for backward compatibility.
All implementation has been split into:

  - vocab_config.py          → Configuration dataclasses
  - vocab_production.py      → SentencePiece-based creation
  - vocab_research.py        → Frequency-based creation
  - vocab_validation.py      → Validation and saving
  - vocabulary_creator.py    → Main orchestrator class

Usage (preferred):
    from vocabulary.vocabulary_creator import UnifiedVocabularyCreator
    from vocabulary.vocab_config import UnifiedVocabConfig, VocabStats, CreationMode, LanguageGroup
"""

from vocabulary_creator import UnifiedVocabularyCreator
from vocab_config import UnifiedVocabConfig, VocabStats, CreationMode, LanguageGroup

__all__ = [
    "UnifiedVocabularyCreator",
    "UnifiedVocabConfig",
    "VocabStats",
    "CreationMode",
    "LanguageGroup",
]
