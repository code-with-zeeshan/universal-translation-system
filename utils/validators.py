# utils/validators.py
import re
from typing import List, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class InputValidator:
    """Validate and sanitize user inputs"""
    
    # Regex patterns
    LANGUAGE_CODE_PATTERN = re.compile(r'^[a-z]{2,3}$', re.IGNORECASE)
    FILENAME_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.]+$')
    MAX_TEXT_LENGTH = 5000
    
    @staticmethod
    def validate_language_code(lang_code: str) -> bool:
        """Validate language code format"""
        if not lang_code or not isinstance(lang_code, str):
            return False
        return bool(InputValidator.LANGUAGE_CODE_PATTERN.match(lang_code.lower()))
    
    @staticmethod
    def validate_text_input(text: str, max_length: Optional[int] = None) -> str:
        """Validate and sanitize text input"""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        
        # Limit length
        max_len = max_length or InputValidator.MAX_TEXT_LENGTH
        if len(text) > max_len:
            logger.warning(f"Text truncated from {len(text)} to {max_len} characters")
            text = text[:max_len]
        
        return text.strip()
    
    @staticmethod
    def validate_filename(filename: str) -> str:
        """Validate filename for safety"""
        if not filename or not isinstance(filename, str):
            raise ValueError("Invalid filename")
        
        # Remove path components
        if Path(filename).is_absolute():
            raise ValueError("Invalid filename: must be relative")
        filename = filename.replace('/', '').replace('\\', '').replace('..', '')
        
        if not InputValidator.FILENAME_PATTERN.match(filename):
            raise ValueError(f"Invalid filename format: {filename}")
        
        return filename
    
    @staticmethod
    def sanitize_path(path_str: str, allowed_dirs: List[str]) -> str:
        """Sanitize and validate file paths"""
        
        path = Path(path_str).resolve()
        
        # Check if path is within allowed directories
        allowed_paths = [Path(d).resolve() for d in allowed_dirs]
        
        if not any(str(path).startswith(str(allowed)) for allowed in allowed_paths):
            raise ValueError(f"Path outside allowed directories: {path}")
        
        return str(path)