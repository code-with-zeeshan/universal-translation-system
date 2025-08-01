# utils/base_classes.py
"""
Base classes for the Universal Translation System
Eliminates code duplication across modules
"""
from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn

class BaseDataProcessor(ABC):
    """Base class for all data processors"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        from utils.common_utils import StandardLogger
        self.logger = logger or StandardLogger.get_logger(self.__class__.__name__)
        
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

class TokenizerMixin:
    """Mixin class for consistent tokenization across modules"""
    
    def tokenize_with_subwords(self, text: str, vocab_pack: Any, language: str) -> List[int]:
        """
        Unified tokenization method with subword handling.
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