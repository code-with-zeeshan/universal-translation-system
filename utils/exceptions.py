# utils/exceptions.py
"""Custom exceptions for the Universal Translation System"""

class TranslationSystemError(Exception):
    """Base exception for translation system"""
    pass

class VocabularyError(TranslationSystemError):
    """Vocabulary-related errors"""
    pass

class TrainingError(TranslationSystemError):
    """Training-related errors"""
    pass

class ModelError(TranslationSystemError):
    """Model-related errors"""
    pass

class DataError(TranslationSystemError):
    """Data processing errors"""
    pass

class ConfigurationError(TranslationSystemError):
    """Configuration errors"""
    pass

# Usage example:
# raise VocabularyError(f"Vocabulary pack not found: {pack_name}")