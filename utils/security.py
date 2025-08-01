# utils/security.py
"""
Enhanced Security utilities for the Universal Translation System
Centralizes all security-related functions
"""
import re
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import logging
import hashlib

logger = logging.getLogger(__name__)

# Trusted model sources
TRUSTED_MODEL_SOURCES = [
    'facebook/',
    'microsoft/', 
    'google/',
    'Helsinki-NLP/',
    'huggingface/',
    'sentence-transformers/',
    'allenai/',
    'OpenNMT/',
    'Unbabel/',
    'xlm-roberta',
    'bert-base',
    'distilbert'
]

def validate_model_source(model_name: str) -> bool:
    """
    Validate model source before loading to prevent security risks.
    
    Args:
        model_name: Name of the model to validate
    
    Returns:
        True if model is from trusted source
    """
    # Check if model is from trusted source
    is_trusted = any(
        source in model_name or model_name.startswith(source) 
        for source in TRUSTED_MODEL_SOURCES
    )
    
    if not is_trusted:
        logger.warning(f"⚠️ Model '{model_name}' is not from a known trusted source")
    
    return is_trusted

def validate_path_component(component: str) -> str:
    """
    Validate path component to prevent path traversal attacks.
    
    Args:
        component: Path component to validate
        
    Returns:
        Validated path component
        
    Raises:
        ValueError: If component contains dangerous characters
    """
    # Check for path traversal attempts
    dangerous_patterns = ['..', '/', '\\', '\x00', '~', '$', '`', '|', ';', '&']
    
    for pattern in dangerous_patterns:
        if pattern in component:
            raise ValueError(f"Invalid path component '{component}': contains '{pattern}'")
    
    # Additional validation using regex
    if not re.match(r'^[a-zA-Z0-9_\-\.]+$', component):
        raise ValueError(f"Invalid path component '{component}': contains invalid characters")
    
    return component

def safe_load_model(model_name: str, model_class=None, **kwargs):
    """
    Safely load a model with security validation.
    
    Args:
        model_name: Name of the model to load
        model_class: Model class to use (e.g., AutoModel)
        **kwargs: Additional arguments for model loading
        
    Returns:
        Loaded model
        
    Raises:
        ValueError: If model source is untrusted
    """
    if not validate_model_source(model_name):
        raise ValueError(f"Untrusted model source: {model_name}")
    
    # Force secure loading
    kwargs['trust_remote_code'] = False
    
    # Add safety checks for local paths
    if '/' in model_name and not any(source in model_name for source in TRUSTED_MODEL_SOURCES):
        # This might be a local path, validate it
        model_path = Path(model_name)
        if not model_path.is_absolute():
            model_path = model_path.resolve()
        
        # Check if path is within allowed directories
        allowed_dirs = [Path.cwd() / 'models', Path.cwd() / 'checkpoints']
        if not any(str(model_path).startswith(str(allowed)) for allowed in allowed_dirs):
            raise ValueError(f"Model path outside allowed directories: {model_path}")
    
    if model_class:
        return model_class.from_pretrained(model_name, **kwargs)
    else:
        # Import here to avoid circular imports
        from transformers import AutoModel
        return AutoModel.from_pretrained(model_name, **kwargs)

def validate_corpus_path(corpus_path: Union[str, Path]) -> Path:
    """
    Validate corpus file path for security.
    
    Args:
        corpus_path: Path to corpus file
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: If path is invalid or outside allowed directories
    """
    corpus_path = Path(corpus_path).resolve()
    
    # Check if path exists
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    
    # Check if path is within allowed directories
    allowed_dirs = [
        Path.cwd() / 'data',
        Path.cwd() / 'corpora',
        Path('/tmp')  # For temporary files
    ]
    
    if not any(str(corpus_path).startswith(str(allowed)) for allowed in allowed_dirs):
        raise ValueError(f"Corpus path outside allowed directories: {corpus_path}")
    
    return corpus_path

def get_file_hash(file_path: Union[str, Path]) -> str:
    """Get SHA256 hash of a file for integrity checking."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize filename for safe file operations.
    
    Args:
        filename: Original filename
        max_length: Maximum allowed length
        
    Returns:
        Sanitized filename
    """
    # Remove path separators and null bytes
    filename = filename.replace('/', '_').replace('\\', '_').replace('\x00', '')
    
    # Remove other dangerous characters
    filename = re.sub(r'[<>:"|?*]', '_', filename)
    
    # Limit length
    if len(filename) > max_length:
        name, ext = Path(filename).stem, Path(filename).suffix
        max_name_length = max_length - len(ext) - 1
        filename = name[:max_name_length] + ext
    
    return filename