"""
Standardized error codes for all components of the Universal Translation System.
This ensures consistent error handling and reporting across all SDKs and components.
"""

from enum import Enum
from typing import Optional, Dict, Any, Type, Union
import traceback
import json

class TranslationErrorCode(Enum):
    """Standardized error codes for all components"""
    NETWORK_ERROR = "NETWORK_ERROR"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    VOCABULARY_NOT_LOADED = "VOCABULARY_NOT_LOADED"
    ENCODING_FAILED = "ENCODING_FAILED"
    DECODING_FAILED = "DECODING_FAILED"
    INVALID_LANGUAGE = "INVALID_LANGUAGE"
    RATE_LIMITED = "RATE_LIMITED"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    TIMEOUT = "TIMEOUT"
    INVALID_INPUT = "INVALID_INPUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    AUTHORIZATION_ERROR = "AUTHORIZATION_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    
    @classmethod
    def from_exception(cls, exception: Exception) -> "TranslationErrorCode":
        """Map exceptions to error codes"""
        message = str(exception).lower()
        
        if "network" in message or "connection" in message or "timeout" in message:
            return cls.NETWORK_ERROR
        elif "model" in message and "not found" in message:
            return cls.MODEL_NOT_FOUND
        elif "vocabulary" in message or "vocab" in message:
            return cls.VOCABULARY_NOT_LOADED
        elif "encoding" in message:
            return cls.ENCODING_FAILED
        elif "decoding" in message:
            return cls.DECODING_FAILED
        elif "language" in message:
            return cls.INVALID_LANGUAGE
        elif "rate" in message and "limit" in message:
            return cls.RATE_LIMITED
        elif "resource" in message or "memory" in message or "gpu" in message:
            return cls.RESOURCE_EXHAUSTED
        elif "timeout" in message:
            return cls.TIMEOUT
        elif "input" in message:
            return cls.INVALID_INPUT
        elif "auth" in message and "fail" in message:
            return cls.AUTHENTICATION_ERROR
        elif "permission" in message or "access" in message:
            return cls.AUTHORIZATION_ERROR
        elif "unavailable" in message or "down" in message:
            return cls.SERVICE_UNAVAILABLE
        else:
            return cls.INTERNAL_ERROR

class TranslationError(Exception):
    """
    Standardized error class for all translation-related errors.
    Includes error code, message, and optional details.
    """
    
    def __init__(
        self, 
        code: Union[TranslationErrorCode, str], 
        message: str, 
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        """
        Initialize a TranslationError.
        
        Args:
            code: Error code (either a TranslationErrorCode enum or a string)
            message: Human-readable error message
            details: Optional dictionary with additional error details
            original_exception: Optional original exception that caused this error
        """
        if isinstance(code, str):
            try:
                self.code = TranslationErrorCode(code)
            except ValueError:
                self.code = TranslationErrorCode.INTERNAL_ERROR
        else:
            self.code = code
            
        self.message = message
        self.details = details or {}
        self.original_exception = original_exception
        
        # Add stack trace if there's an original exception
        if original_exception:
            self.details["original_error"] = str(original_exception)
            self.details["stack_trace"] = traceback.format_exc()
        
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization"""
        return {
            "code": self.code.value,
            "message": self.message,
            "details": self.details
        }
    
    def to_json(self) -> str:
        """Convert error to JSON string"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_exception(cls, exception: Exception, default_message: str = "An error occurred") -> "TranslationError":
        """Create a TranslationError from an exception"""
        code = TranslationErrorCode.from_exception(exception)
        message = str(exception) or default_message
        return cls(code, message, original_exception=exception)

# Error handling utilities
def safe_execute(func, *args, default_value=None, error_code=TranslationErrorCode.INTERNAL_ERROR, **kwargs):
    """
    Execute a function safely, catching and standardizing any exceptions.
    
    Args:
        func: Function to execute
        *args: Arguments to pass to the function
        default_value: Value to return if an exception occurs
        error_code: Error code to use if an exception occurs
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Result of the function or default_value if an exception occurs
        
    Raises:
        TranslationError: Standardized error if an exception occurs
    """
    try:
        return func(*args, **kwargs)
    except TranslationError as e:
        # Pass through TranslationError instances
        raise e
    except Exception as e:
        # Convert other exceptions to TranslationError
        raise TranslationError(error_code, str(e), original_exception=e)

# HTTP status code mapping
HTTP_STATUS_CODES = {
    TranslationErrorCode.NETWORK_ERROR: 503,
    TranslationErrorCode.MODEL_NOT_FOUND: 404,
    TranslationErrorCode.VOCABULARY_NOT_LOADED: 404,
    TranslationErrorCode.ENCODING_FAILED: 500,
    TranslationErrorCode.DECODING_FAILED: 500,
    TranslationErrorCode.INVALID_LANGUAGE: 400,
    TranslationErrorCode.RATE_LIMITED: 429,
    TranslationErrorCode.RESOURCE_EXHAUSTED: 503,
    TranslationErrorCode.TIMEOUT: 504,
    TranslationErrorCode.INVALID_INPUT: 400,
    TranslationErrorCode.INTERNAL_ERROR: 500,
    TranslationErrorCode.AUTHENTICATION_ERROR: 401,
    TranslationErrorCode.AUTHORIZATION_ERROR: 403,
    TranslationErrorCode.SERVICE_UNAVAILABLE: 503,
}