# utils/base_classes.py
"""
Base classes for the Universal Translation System
Eliminates code duplication across modules
"""
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import threading
from typing import Dict, List, Optional, Any, TypeVar, Generic, Callable, Union
import torch
import torch.nn as nn
import os
from config.schemas import RootConfig

T = TypeVar('T')
R = TypeVar('R')

class BaseManager(ABC):
    """
    Base class for all resource managers.
    Provides common functionality for resource management.
    
    Thread Safety:
        All methods are thread-safe unless explicitly noted.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the manager.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._lock = threading.RLock()
        self._initialized = False
        
    def initialize(self) -> None:
        """
        Initialize the manager.
        This method is idempotent and thread-safe.
        """
        with self._lock:
            if not self._initialized:
                self._do_initialize()
                self._initialized = True
                
    @abstractmethod
    def _do_initialize(self) -> None:
        """
        Perform actual initialization.
        Must be implemented by subclasses.
        """
        pass
        
    def shutdown(self) -> None:
        """
        Shutdown the manager.
        This method is idempotent and thread-safe.
        """
        with self._lock:
            if self._initialized:
                self._do_shutdown()
                self._initialized = False
                
    @abstractmethod
    def _do_shutdown(self) -> None:
        """
        Perform actual shutdown.
        Must be implemented by subclasses.
        """
        pass
        
    def __enter__(self):
        """Context manager support."""
        self.initialize()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support."""
        self.shutdown()


class BaseProcessor(Generic[T, R]):
    """
    Base class for all data processors.
    Provides common functionality for data processing.
    
    Thread Safety:
        All methods are thread-safe unless explicitly noted.
    
    Type Parameters:
        T: Input type
        R: Result type
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the processor.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._lock = threading.RLock()
        self._stats = {
            "processed_items": 0,
            "failed_items": 0,
            "processing_time": 0.0
        }
        
    @abstractmethod
    def process(self, item: T) -> R:
        """
        Process a single item.
        Must be implemented by subclasses.
        
        Args:
            item: Item to process
            
        Returns:
            Processed result
        """
        pass
        
    def process_batch(self, items: List[T]) -> List[R]:
        """
        Process a batch of items.
        
        Args:
            items: Items to process
            
        Returns:
            Processed results
        """
        results = []
        for item in items:
            try:
                result = self.process(item)
                results.append(result)
                with self._lock:
                    self._stats["processed_items"] += 1
            except Exception as e:
                self.logger.error(f"Error processing item: {e}")
                with self._lock:
                    self._stats["failed_items"] += 1
                raise
        return results
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get processor statistics.
        
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            return self._stats.copy()


class BaseValidator(ABC):
    """
    Base class for all validators.
    Provides common functionality for data validation.
    
    Thread Safety:
        All methods are thread-safe unless explicitly noted.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the validator.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def validate(self, data: Any) -> bool:
        """
        Validate data.
        Must be implemented by subclasses.
        
        Args:
            data: Data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        pass
        
    def validate_or_raise(self, data: Any, error_message: str = "Validation failed") -> Any:
        """
        Validate data and raise an exception if invalid.
        
        Args:
            data: Data to validate
            error_message: Error message if validation fails
            
        Returns:
            The validated data
            
        Raises:
            ValidationError: If validation fails
        """
        from utils.exceptions import ValidationError
        
        if not self.validate(data):
            raise ValidationError(error_message)
        return data


class TokenizerMixin:
    """
    Mixin for tokenization functionality.
    Provides common methods for tokenizing text.
    
    Thread Safety:
        All methods are thread-safe if the implementing class uses proper locking.
    """
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into tokens.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Default implementation uses simple whitespace tokenization
        return text.split()
        
    def detokenize(self, tokens: List[str]) -> str:
        """
        Detokenize tokens into text.
        
        Args:
            tokens: Tokens to detokenize
            
        Returns:
            Detokenized text
        """
        # Default implementation uses simple whitespace joining
        return " ".join(tokens)
        
    def encode(self, text: str) -> List[int]:
        """
        Encode text into token IDs.
        
        Args:
            text: Text to encode
            
        Returns:
            List of token IDs
        """
        tokens = self.tokenize(text)
        return self.tokens_to_ids(tokens)
        
    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs into text.
        
        Args:
            ids: Token IDs to decode
            
        Returns:
            Decoded text
        """
        tokens = self.ids_to_tokens(ids)
        return self.detokenize(tokens)
        
    @abstractmethod
    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        Convert tokens to token IDs.
        Must be implemented by subclasses.
        
        Args:
            tokens: Tokens to convert
            
        Returns:
            List of token IDs
        """
        pass
        
    @abstractmethod
    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        """
        Convert token IDs to tokens.
        Must be implemented by subclasses.
        
        Args:
            ids: Token IDs to convert
            
        Returns:
            List of tokens
        """
        pass

    def tokenize_with_subwords(self, text: str, vocab_pack: Any, language: str) -> List[int]:
        """Unified tokenization method with subword handling.
        Used across all modules to ensure consistency.
        """
        tokens = []
        
        # Add language/start token
        start_token = vocab_pack.special_tokens.get(f'<{language}>', 
                                                    vocab_pack.special_tokens.get('<s>', 2))
        tokens.append(start_token)
        
        # Tokenize text
        words = text.lower().split()
        
        for word in words:
            if word in vocab_pack.tokens:
                tokens.append(vocab_pack.tokens[word])
            else:
                # Subword tokenization
                subword_tokens = self._get_subword_tokens(word, vocab_pack)
                tokens.extend(subword_tokens)
        
        # Add end token
        tokens.append(vocab_pack.special_tokens.get('</s>', 3))
        
        return tokens
    
    def _get_subword_tokens(self, word: str, vocab_pack: Any) -> List[int]:
        """Get subword tokens for out-of-vocabulary words"""
        subword_tokens = []
        
        # Implementation of subword tokenization
        # This centralizes the logic used across multiple files
        if hasattr(vocab_pack, 'subwords'):
            i = 0
            while i < len(word):
                matched = False
                for j in range(len(word), i, -1):
                    subword = word[i:j]
                    
                    if i == 0:
                        if subword in vocab_pack.tokens:
                            subword_tokens.append(vocab_pack.tokens[subword])
                            i = j
                            matched = True
                            break
                    else:
                        subword_with_prefix = f"##{subword}"
                        if subword_with_prefix in vocab_pack.subwords:
                            subword_tokens.append(vocab_pack.subwords[subword_with_prefix])
                            i = j
                            matched = True
                            break

                if not matched:
                    i += 1  # Skip character if no match            
        
        # Fallback to UNK token
        if not subword_tokens:
            subword_tokens.append(vocab_pack.special_tokens.get('<unk>', 1))
        
        return subword_tokens

class BaseDataProcessor(ABC):
    """Base class for all data processors"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def process(self, *args, **kwargs):
        """Process data - must be implemented by subclasses"""
        pass
    
    def validate_input(self, data: Any) -> bool:
        """Common input validation"""
        if data is None:
            self.logger.error("Input data is None")
            return False
        return True

class BaseVocabularyManager(ABC):
    """Base class for vocabulary managers."""

    def __init__(self, config: RootConfig, vocab_dir: str = 'vocabs'):
        self.config = config
        self.vocab_dir = Path(vocab_dir)
        self.language_to_pack = self.config.vocabulary.language_to_pack_mapping
        self.logger = logging.getLogger(self.__class__.__name__)
        self._validate_vocab_dir()

    def _validate_vocab_dir(self):
        """Validate vocabulary directory exists and is accessible"""
        if not self.vocab_dir.exists():
            raise FileNotFoundError(f"Vocabulary directory not found: {self.vocab_dir}")
        if not os.access(self.vocab_dir, os.R_OK):
            raise PermissionError(f"Cannot read vocabulary directory: {self.vocab_dir}")

    @abstractmethod
    def get_vocab_for_pair(self, source_lang: str, target_lang: str, version: Optional[str] = None) -> Any:
        """Get appropriate vocabulary pack for a language pair."""
        pass

class BaseVocabularyHandler(ABC):
    """Base class for vocabulary-related operations"""
    
    def __init__(self, vocab_dir: str = 'vocabs'):
        self.vocab_dir = Path(vocab_dir)
        self._validate_vocab_dir()
    
    def _validate_vocab_dir(self):
        """Validate vocabulary directory exists and is accessible"""
        if not self.vocab_dir.exists():
            raise FileNotFoundError(f"Vocabulary directory not found: {self.vocab_dir}")
        if not os.access(self.vocab_dir, os.R_OK):
            raise PermissionError(f"Cannot read vocabulary directory: {self.vocab_dir}")
    
    @abstractmethod
    def load_vocabulary(self, *args, **kwargs):
        """Load vocabulary - must be implemented by subclasses"""
        pass

