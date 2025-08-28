# data/dataset_classes.py
"""
Dataset classes for the Universal Translation System
"""
import torch
from torch.utils.data import Dataset
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
from utils.base_classes import TokenizerMixin

logger = logging.getLogger(__name__)


class ModernParallelDataset(Dataset, TokenizerMixin):
    """
    Modern parallel dataset with caching and preprocessing.
    Extracted from train_universal_system.py
    """
    
    def __init__(self, data_path: str, cache_dir: Optional[str] = None, vocab_dir: str = 'vocabs', config: Optional[RootConfig] = None):
        self.data_path = Path(data_path)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_path.parent / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load or create cached data
        self.data = self._load_or_create_cache()
        
        # Initialize VocabularyManager
        from vocabulary.unified_vocab_manager import UnifiedVocabularyManager, VocabularyMode
        from config.schemas import load_config as load_pydantic_config
        
        # Ensure we have a valid config for the vocab manager
        self.config = config or load_pydantic_config()
        
        # Use OPTIMIZED mode for dataset processing
        self.vocab_manager = UnifiedVocabularyManager(config=self.config, vocab_dir=vocab_dir, mode=VocabularyMode.OPTIMIZED)
        
        logger.info(f"📚 Dataset loaded: {len(self.data)} samples")
    
    def _load_or_create_cache(self):
        """Load cached data or create cache from raw data"""
        
        cache_file = self.cache_dir / f"{self.data_path.stem}_cache.json"
        
        if cache_file.exists():
            logger.info(f"📦 Loading cached data from {cache_file}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        logger.info(f"🔄 Creating cache from {self.data_path}")
        data = self._load_raw_data()
        
        # Save cache
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return data
    
    def _load_raw_data(self):
        """Load and preprocess raw parallel data"""
        
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                try:
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        data.append({
                            'source': parts[0].strip(),
                            'target': parts[1].strip(),
                            'source_lang': parts[2].strip(),
                            'target_lang': parts[3].strip(),
                            'line_no': line_no
                        })
                except Exception as e:
                    logger.warning(f"⚠️ Error processing line {line_no}: {e}")
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get vocabulary pack for this language pair
        vocab_pack = self.vocab_manager.get_vocab_for_pair(
            item['source_lang'], 
            item['target_lang']
        )
        
        # Tokenize with modern preprocessing (using inherited TokenizerMixin method)
        source_tokens = self.tokenize_with_subwords(
            item['source'], 
            vocab_pack,
            item['source_lang']
        )
        target_tokens = self.tokenize_with_subwords(
            item['target'], 
            vocab_pack,
            item['target_lang']
        )
        
        # Pad or truncate to max length (e.g., 512)
        max_length = 512
        source_tokens = self._pad_or_truncate(source_tokens, max_length)
        target_tokens = self._pad_or_truncate(target_tokens, max_length)
        
        # Create attention masks
        source_mask = [1 if tok != 0 else 0 for tok in source_tokens]
        target_mask = [1 if tok != 0 else 0 for tok in target_tokens]
        
        return {
            'source_ids': torch.tensor(source_tokens, dtype=torch.long),
            'target_ids': torch.tensor(target_tokens, dtype=torch.long),
            'source_mask': torch.tensor(source_mask, dtype=torch.long),
            'target_mask': torch.tensor(target_mask, dtype=torch.long),
            # Add vocab_pack info
            'vocab_pack_name': vocab_pack.name if hasattr(vocab_pack, 'name') else 'default',
            'vocab_size': vocab_pack.size if hasattr(vocab_pack, 'size') else len(vocab_pack.tokens),
            'pad_token_id': vocab_pack.special_tokens.get('<pad>', 0), 
            'unk_token_id': vocab_pack.special_tokens.get('<unk>', 1),
            'metadata': {
                'source_lang': item['source_lang'],
                'target_lang': item['target_lang'],
                'line_no': item.get('line_no', idx)
            }
        }
    
    def _pad_or_truncate(self, tokens: List[int], max_length: int) -> List[int]:
        """Pad or truncate tokens to max_length"""
        if len(tokens) > max_length:
            return tokens[:max_length]
        else:
            return tokens + [0] * (max_length - len(tokens))