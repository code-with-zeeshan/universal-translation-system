# utils/security.py
"""
Security utilities for the Universal Translation System
"""
import os
import re
from pathlib import Path
from typing import List, Optional
import logging
from .exceptions import SecurityError

logger = logging.getLogger(__name__)

# Trusted model sources
TRUSTED_SOURCES = [
    'facebook/',
    'microsoft/',
    'google/',
    'Helsinki-NLP/',
    'huggingface/',
    'openai/',
    'anthropic/'
]

def validate_path_component(path_component: str) -> str:
    """
    Validate and sanitize a path component to prevent path traversal attacks.
    
    Args:
        path_component: The path component to validate
        
    Returns:
        Sanitized path component
        
    Raises:
        SecurityError: If the path component is invalid or potentially dangerous
    """
    if not path_component:
        raise SecurityError("Path component cannot be empty")
    
    # Check for path traversal attempts
    dangerous_patterns = [
        '..',
        '/',
        '\\',
        ':',
        '<',
        '>',
        '|',
        '?',
        '*'
    ]
    
    for pattern in dangerous_patterns:
        if pattern in path_component:
            raise SecurityError(f"Invalid character '{pattern}' in path component: {path_component}")
    
    # Check for null bytes
    if '\x00' in path_component:
        raise SecurityError("Null byte detected in path component")
    
    # Sanitize the component
    sanitized = re.sub(r'[^\w\-_.]', '_', path_component)
    
    # Ensure it's not too long
    if len(sanitized) > 255:
        raise SecurityError("Path component too long")
    
    return sanitized

def validate_file_path(file_path: str, allowed_directories: Optional[List[str]] = None) -> Path:
    """
    Validate a file path for security.
    
    Args:
        file_path: The file path to validate
        allowed_directories: List of allowed base directories
        
    Returns:
        Validated Path object
        
    Raises:
        SecurityError: If the path is invalid or outside allowed directories
    """
    try:
        path = Path(file_path).resolve()
    except Exception as e:
        raise SecurityError(f"Invalid file path: {e}")
    
    # Check if path exists and is a file
    if not path.exists():
        raise SecurityError(f"File does not exist: {path}")
    
    if not path.is_file():
        raise SecurityError(f"Path is not a file: {path}")
    
    # Check allowed directories if specified
    if allowed_directories:
        allowed = False
        for allowed_dir in allowed_directories:
            try:
                allowed_path = Path(allowed_dir).resolve()
                # Python 3.8-compatible check: try relative_to and catch ValueError
                try:
                    path.relative_to(allowed_path)
                    allowed = True
                    break
                except ValueError:
                    continue
            except Exception:
                continue
        
        if not allowed:
            raise SecurityError(f"File path not in allowed directories: {path}")
    
    return path

def validate_model_source(model_name: str) -> bool:
    """
    Validate if a model comes from a trusted source.
    
    Args:
        model_name: Name of the model (e.g., 'facebook/nllb-200-distilled-600M')
        
    Returns:
        True if from trusted source, False otherwise
    """
    if not model_name:
        return False
    
    for trusted_source in TRUSTED_SOURCES:
        if model_name.startswith(trusted_source):
            return True
    
    logger.warning(f"Model from untrusted source: {model_name}")
    return False

def check_file_size(file_path: str, max_size_gb: float = 10.0) -> bool:
    """
    Check if file size is within acceptable limits.
    
    Args:
        file_path: Path to the file
        max_size_gb: Maximum allowed size in GB
        
    Returns:
        True if file size is acceptable, False otherwise
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return False
        
        size_gb = path.stat().st_size / (1024 ** 3)
        return size_gb <= max_size_gb
    
    except Exception as e:
        logger.error(f"Error checking file size: {e}")
        return False

def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to make it safe for filesystem operations.
    
    Args:
        filename: The filename to sanitize
        
    Returns:
        Sanitized filename
    """
    # Remove or replace dangerous characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove control characters
    sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)
    
    # Limit length
    if len(sanitized) > 255:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:255-len(ext)] + ext
    
    # Ensure it's not empty
    if not sanitized.strip():
        sanitized = "unnamed_file"
    
    return sanitized

def validate_config_value(value: str, allowed_values: List[str]) -> bool:
    """
    Validate that a configuration value is in the allowed list.
    
    Args:
        value: The value to validate
        allowed_values: List of allowed values
        
    Returns:
        True if value is allowed, False otherwise
    """
    return value in allowed_values

class SecurityContext:
    """Context manager for security-sensitive operations"""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        logger.info(f"Starting security-sensitive operation: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.time() - self.start_time
        
        if exc_type:
            logger.error(f"Security operation failed: {self.operation_name} ({duration:.2f}s)")
        else:
            logger.info(f"Security operation completed: {self.operation_name} ({duration:.2f}s)")
        
        return False  # Don't suppress exceptions