# utils/dataset_classes.py
"""
Complete dataset classes used across the training system
"""
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
import json
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class TranslationSample:
    """Single translation sample"""
    source_text: str
    target_text: str
    source_lang: str
    target_lang: str
    
class ModernParallelDataset(Dataset):
    """Modern parallel dataset for training with full implementation"""
    
    def __init__(self, 
                 data_path: str,
                 vocab_manager: Optional[Any] = None,
                 max_length: int = 512,
                 cache_size: int = 10000,
                 tokenizer: Optional[Any] = None):
        self.data_path = Path(data_path)
        self.vocab_manager = vocab_manager
        self.max_length = max_length
        self.cache_size = cache_size
        self.tokenizer = tokenizer
        
        # Load data
        self.samples = self._load_data()
        
        # Cache for tokenized samples
        self._cache = {}
        self._cache_order = []
        
        logger.info(f"Loaded {len(self.samples)} samples from {data_path}")
    
    def _load_data(self) -> List[TranslationSample]:
        """Load data from file"""
        samples = []
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    samples.append(TranslationSample(
                        source_text=parts[0],
                        target_text=parts[1],
                        source_lang=parts[2],
                        target_lang=parts[3]
                    ))
                elif len(parts) == 2:
                    # Assume en->es if no language specified
                    samples.append(TranslationSample(
                        source_text=parts[0],
                        target_text=parts[1],
                        source_lang='en',
                        target_lang='es'
                    ))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Check cache
        if idx in self._cache:
            return self._cache[idx]
        
        sample = self.samples[idx]
        
        # Tokenize if vocab_manager is available
        if self.vocab_manager:
            vocab_pack = self.vocab_manager.get_vocab_for_pair(
                sample.source_lang, sample.target_lang
            )
            
            # Tokenize source
            source_tokens = self._tokenize_text(
                sample.source_text, sample.source_lang, vocab_pack
            )
            
            # Tokenize target
            target_tokens = self._tokenize_text(
                sample.target_text, sample.target_lang, vocab_pack
            )
            
            # Truncate or pad
            source_ids = self._prepare_tokens(source_tokens, self.max_length)
            target_ids = self._prepare_tokens(target_tokens, self.max_length)
            
            # Create attention mask
            source_mask = [1 if tok != 0 else 0 for tok in source_ids]
            
            item = {
                'source_ids': torch.tensor(source_ids, dtype=torch.long),
                'target_ids': torch.tensor(target_ids, dtype=torch.long),
                'source_mask': torch.tensor(source_mask, dtype=torch.long),
                'metadata': {
                    'source_lang': sample.source_lang,
                    'target_lang': sample.target_lang
                },
                'vocab_pack': vocab_pack
            }
        else:
            # Return raw text if no vocab_manager
            item = {
                'source_text': sample.source_text,
                'target_text': sample.target_text,
                'source_ids': torch.zeros(self.max_length, dtype=torch.long),
                'target_ids': torch.zeros(self.max_length, dtype=torch.long),
                'source_mask': torch.zeros(self.max_length, dtype=torch.long),
                'metadata': {
                    'source_lang': sample.source_lang,
                    'target_lang': sample.target_lang
                }
            }
        
        # Add to cache
        if len(self._cache) < self.cache_size:
            self._cache[idx] = item
            self._cache_order.append(idx)
        else:
            # LRU eviction
            old_idx = self._cache_order.pop(0)
            del self._cache[old_idx]
            self._cache[idx] = item
            self._cache_order.append(idx)
        
        return item
    
    def _tokenize_text(self, text: str, language: str, vocab_pack: Any) -> List[int]:
        """Tokenize text using vocabulary pack"""
        tokens = []
        
        # Add start token
        start_token = vocab_pack.special_tokens.get(f'<{language}>', 
                                                    vocab_pack.special_tokens.get('<s>', 2))
        tokens.append(start_token)
        
        # Simple whitespace tokenization
        words = text.lower().split()
        
        for word in words:
            if word in vocab_pack.tokens:
                tokens.append(vocab_pack.tokens[word])
            else:
                # Handle unknown words
                if hasattr(vocab_pack, 'subwords'):
                    # Try subword tokenization
                    subword_tokens = self._get_subword_tokens(word, vocab_pack)
                    tokens.extend(subword_tokens)
                else:
                    # Use UNK token
                    tokens.append(vocab_pack.special_tokens.get('<unk>', 1))
        
        # Add end token
        tokens.append(vocab_pack.special_tokens.get('</s>', 3))
        
        return tokens
    
    def _get_subword_tokens(self, word: str, vocab_pack: Any) -> List[int]:
        """Get subword tokens for out-of-vocabulary words"""
        subword_tokens = []
        
        # Simple subword tokenization
        i = 0
        while i < len(word):
            matched = False
            for j in range(len(word), i, -1):
                subword = word[i:j]
                if i > 0:
                    subword = f"##{subword}"
                
                if subword in vocab_pack.subwords:
                    subword_tokens.append(vocab_pack.subwords[subword])
                    i = j
                    matched = True
                    break
            if not matched:
                i += 1
        
        if not subword_tokens:
            subword_tokens.append(vocab_pack.special_tokens.get('<unk>', 1))
        
        return subword_tokens
    
    def _prepare_tokens(self, tokens: List[int], max_length: int) -> List[int]:
        """Truncate or pad tokens to max_length"""
        if len(tokens) > max_length:
            # Truncate
            return tokens[:max_length]
        else:
            # Pad with zeros
            return tokens + [0] * (max_length - len(tokens))

def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for batching"""
    # Stack tensors
    source_ids = torch.stack([item['source_ids'] for item in batch])
    target_ids = torch.stack([item['target_ids'] for item in batch])
    source_mask = torch.stack([item['source_mask'] for item in batch])
    
    # Get metadata
    metadata = {
        'source_lang': [item['metadata']['source_lang'] for item in batch],
        'target_lang': [item['metadata']['target_lang'] for item in batch]
    }
    
    # Get vocab pack (assuming same for batch)
    vocab_pack = batch[0].get('vocab_pack')
    
    return {
        'source_ids': source_ids,
        'target_ids': target_ids,
        'source_mask': source_mask,
        'metadata': metadata,
        'vocab_pack': vocab_pack
    }