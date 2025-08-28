# utils/error_handler.py
"""
Centralized error handling for the Universal Translation System.
This module provides a standardized way to handle and report errors.
"""

import logging
import traceback
import sys
from typing import Callable, TypeVar, Any, Dict, Optional, Type, Union
from functools import wraps
from .exceptions import UniversalTranslationError, ConfigurationError

logger = logging.getLogger(__name__)

T = TypeVar('T')
ErrorHandler = Callable[[Exception], Any]


class ErrorHandlingContext:
    """Context manager for standardized error handling."""
    
    def __init__(
        self, 
        operation_name: str, 
        error_handlers: Optional[Dict[Type[Exception], ErrorHandler]] = None,
        default_handler: Optional[ErrorHandler] = None,
        reraise: bool = True,
        log_level: int = logging.ERROR
    ):
        """
        Initialize error handling context.
        
        Args:
            operation_name: Name of the operation for logging
            error_handlers: Dictionary mapping exception types to handler functions
            default_handler: Handler for exceptions not in error_handlers
            reraise: Whether to reraise exceptions after handling
            log_level: Logging level for errors
        """
        self.operation_name = operation_name
        self.error_handlers = error_handlers or {}
        self.default_handler = default_handler
        self.reraise = reraise
        self.log_level = log_level
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return True
            
        # Find the most specific handler
        handler = None
        for exception_type, handler_func in self.error_handlers.items():
            if issubclass(exc_type, exception_type):
                handler = handler_func
                break
                
        # Use default handler if no specific handler found
        if handler is None:
            handler = self.default_handler
            
        # Log the error
        logger.log(
            self.log_level,
            f"Error in {self.operation_name}: {exc_val}",
            exc_info=(exc_type, exc_val, exc_tb)
        )
        
        # Handle the error
        if handler:
            try:
                handler(exc_val)
            except Exception as e:
                logger.error(f"Error in error handler: {e}")
                
        # Return True to suppress the exception, False to reraise
        return not self.reraise


def handle_errors(
    operation_name: Optional[str] = None,
    error_handlers: Optional[Dict[Type[Exception], ErrorHandler]] = None,
    default_handler: Optional[ErrorHandler] = None,
    reraise: bool = True,
    log_level: int = logging.ERROR
):
    """
    Decorator for standardized error handling.
    
    Args:
        operation_name: Name of the operation for logging
        error_handlers: Dictionary mapping exception types to handler functions
        default_handler: Handler for exceptions not in error_handlers
        reraise: Whether to reraise exceptions after handling
        log_level: Logging level for errors
        
    Returns:
        Decorated function
    """
    def decorator(func):
        nonlocal operation_name
        if operation_name is None:
            operation_name = func.__name__
            
        @wraps(func)
        def wrapper(*args, **kwargs):
            with ErrorHandlingContext(
                operation_name,
                error_handlers,
                default_handler,
                reraise,
                log_level
            ):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def default_error_response(error: Exception) -> Dict[str, Any]:
    """
    Create a standardized error response.
    
    Args:
        error: The exception
        
    Returns:
        Error response dictionary
    """
    error_type = type(error).__name__
    
    # Get error code for UniversalTranslationError
    error_code = getattr(error, 'code', None)
    if error_code is None and isinstance(error, UniversalTranslationError):
        error_code = f"UTS-{error_type}"
    
    response = {
        "success": False,
        "error": {
            "type": error_type,
            "message": str(error)
        }
    }
    
    if error_code:
        response["error"]["code"] = error_code
        
    # Add details for ConfigurationError
    if isinstance(error, ConfigurationError):
        response["error"]["details"] = getattr(error, 'details', {})
        
    return response


def log_exception(
    exc_info=None, 
    level: int = logging.ERROR, 
    logger_name: Optional[str] = None
) -> None:
    """
    Log an exception with standardized formatting.
    
    Args:
        exc_info: Exception info tuple (type, value, traceback)
        level: Logging level
        logger_name: Logger name (defaults to current module)
    """
    if exc_info is None:
        exc_info = sys.exc_info()
        
    exc_type, exc_value, exc_traceback = exc_info
    
    if not exc_type:
        return
        
    log = logger
    if logger_name:
        log = logging.getLogger(logger_name)
        
    # Format traceback
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    tb_text = ''.join(tb_lines)
    
    # Log the error
    log.log(level, f"Exception: {exc_value}\n{tb_text}")


# Register standard error handlers
standard_error_handlers = {
    ConfigurationError: lambda e: logger.error(f"Configuration error: {e}"),
    ValueError: lambda e: logger.warning(f"Value error: {e}"),
    KeyError: lambda e: logger.warning(f"Key error: {e}"),
    FileNotFoundError: lambda e: logger.error(f"File not found: {e}"),
    PermissionError: lambda e: logger.error(f"Permission error: {e}"),
    TimeoutError: lambda e: logger.error(f"Timeout error: {e}"),
    ConnectionError: lambda e: logger.error(f"Connection error: {e}")
}