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

class LoggingError(UniversalTranslationError):
    """Raised when there are logging-related issues"""
    pass
class NetworkError(UniversalTranslationError):
    """Raised when there are network-related issues"""
    pass

class TimeoutError(UniversalTranslationError):
    """Raised when an operation times out"""
    pass

class AuthenticationError(UniversalTranslationError):
    """Raised when there are authentication issues"""
    pass

class AuthorizationError(UniversalTranslationError):
    """Raised when there are authorization issues"""
    pass

class ResourceError(UniversalTranslationError):
    """Raised when there are issues with resource management"""
    pass

class MemoryError(ResourceError):
    """Raised when there are memory-related issues"""
    pass

class ThreadingError(UniversalTranslationError):
    """Raised when there are threading-related issues"""
    pass

class ValidationError(UniversalTranslationError):
    """Raised when validation fails"""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.details = details or {}
