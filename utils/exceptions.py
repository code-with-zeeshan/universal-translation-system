# utils/exceptions.py
"""
Custom exceptions for the Universal Translation System
"""

class UniversalTranslationError(Exception):
    """Base exception for all universal translation system errors"""
    pass

class DataError(UniversalTranslationError):
    """Raised when there are issues with data processing"""
    pass

class VocabularyError(UniversalTranslationError):
    """Raised when there are issues with vocabulary management"""
    pass

class ModelError(UniversalTranslationError):
    """Raised when there are issues with model operations"""
    pass

class ConfigurationError(UniversalTranslationError):
    """Raised when there are configuration issues"""
    pass

class TrainingError(UniversalTranslationError):
    """Raised when there are training-related issues"""
    pass

class InferenceError(UniversalTranslationError):
    """Raised when there are inference-related issues"""
    pass

class ResourceError(UniversalTranslationError):
    """Raised when there are resource-related issues (memory, disk, etc.)"""
    pass

class SecurityError(UniversalTranslationError):
    """Raised when there are security-related issues"""
    pass