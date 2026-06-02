# data/dataset_classes.py
"""
Dataset classes for the Universal Translation System
"""
import torch
from torch.utils.data import Dataset
import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any
from utils.base_classes import TokenizerMixin
from config.schemas import RootConfig, load_config as load_pydantic_config

logger = logging.getLogger(__name__)


class ModernParallelDataset(Dataset, TokenizerMixin):
    """
    Modern parallel dataset with pre-tokenized cache via memmap.
    Tokenizes all samples once on first run; subsequent runs and DataLoader
    workers share the OS page cache via mmap, eliminating per-process copies.
    """

    def __init__(self, data_path: str, cache_dir: Optional[str] = None, vocab_dir: str = 'vocabulary/vocab', config: Optional[RootConfig] = None):
        self.data_path = Path(data_path)
        self.config = config or load_pydantic_config()
        self.max_length = self.config.model.max_seq_length
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_path.parent / "cache"
        self.cache_dir.mkdir(exist_ok=True)

        self._load_or_build_token_cache(vocab_dir)

    def _cache_path(self, suffix: str) -> str:
        return str(self.cache_dir / f"{self.data_path.stem}_tokens_ml{self.max_length}_{suffix}")

    def _load_or_build_token_cache(self, vocab_dir: str):
        src_path = self._cache_path('source_ids.npy')
        if Path(src_path).exists():
            self._load_token_cache()
        else:
            self._build_token_cache(vocab_dir)

    def _load_token_cache(self):
        if not hasattr(self, '_metadata'):
            import pickle
            pkl_path = self._cache_path('metadata.pkl')
            if Path(pkl_path).exists():
                with open(pkl_path, 'rb') as f:
                    self._metadata = pickle.load(f)
            else:
                with open(self._cache_path('metadata.json'), 'r') as f:
                    self._metadata = json.load(f)
        N = len(self._metadata)
        self._num_samples = N

        self._src_ids = np.memmap(self._cache_path('source_ids.npy'), dtype=np.uint16, mode='r', shape=(N, self.max_length))
        self._tgt_ids = np.memmap(self._cache_path('target_ids.npy'), dtype=np.uint16, mode='r', shape=(N, self.max_length))
        self._src_mask = np.memmap(self._cache_path('source_mask.npy'), dtype=np.bool_, mode='r', shape=(N, self.max_length))
        self._tgt_mask = np.memmap(self._cache_path('target_mask.npy'), dtype=np.bool_, mode='r', shape=(N, self.max_length))

        logger.info(f"📚 Pre-tokenized cache loaded: {N} samples ({self._cache_size_gb():.1f}GB shared via mmap)")

    def _cache_size_gb(self) -> float:
        total = 0
        for suffix in ('source_ids.npy', 'target_ids.npy', 'source_mask.npy', 'target_mask.npy'):
            p = Path(self._cache_path(suffix))
            if p.exists():
                total += p.stat().st_size
        return total / (1024**3)

    def _build_token_cache(self, vocab_dir: str):
        items = self._load_or_create_cache()
        N = len(items)
        logger.info(f"🔄 Pre-tokenizing {N} samples (first run, ~10-20 min)...")

        from vocabulary.unified_vocab_manager import UnifiedVocabularyManager, VocabularyMode
        vocab_mgr = UnifiedVocabularyManager(config=self.config, vocab_dir=vocab_dir, mode=VocabularyMode.OPTIMIZED)
        if self.config:
            vocab_mgr.preload_for_languages(list(self.config.data.active_languages))

        src_ids = np.memmap(self._cache_path('source_ids.npy'), dtype=np.uint16, mode='w+', shape=(N, self.max_length))
        tgt_ids = np.memmap(self._cache_path('target_ids.npy'), dtype=np.uint16, mode='w+', shape=(N, self.max_length))
        src_mask = np.memmap(self._cache_path('source_mask.npy'), dtype=np.bool_, mode='w+', shape=(N, self.max_length))
        tgt_mask = np.memmap(self._cache_path('target_mask.npy'), dtype=np.bool_, mode='w+', shape=(N, self.max_length))
        metadata = []

        try:
            from tqdm import tqdm
            iterator = tqdm(items, desc="Tokenizing")
        except ImportError:
            iterator = items

        for i, item in enumerate(iterator):
            vocab_pack = vocab_mgr.get_vocab_for_pair(item['source_lang'], item['target_lang'])
            src_tokens = self.tokenize_with_subwords(item['source'], vocab_pack, item['source_lang'])
            tgt_tokens = self.tokenize_with_subwords(item['target'], vocab_pack, item['target_lang'])

            src = self._pad_or_truncate(src_tokens, self.max_length)
            tgt = self._pad_or_truncate(tgt_tokens, self.max_length)

            src_ids[i] = src
            tgt_ids[i] = tgt
            src_mask[i] = [t != 0 for t in src]
            tgt_mask[i] = [t != 0 for t in tgt]
            metadata.append({
                'source_lang': item['source_lang'],
                'target_lang': item['target_lang'],
                'line_no': item.get('line_no', i),
            })

        src_ids.flush()
        del src_ids, tgt_ids, src_mask, tgt_mask, items, vocab_mgr

        import pickle
        with open(self._cache_path('metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)

        self._load_token_cache()

    def _load_or_create_cache(self):
        cache_file = self.cache_dir / f"{self.data_path.stem}_cache.json"
        if cache_file.exists():
            logger.info(f"📦 Loading raw data from {cache_file}")
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Cache corrupted ({e}), regenerating...")
                cache_file.unlink(missing_ok=True)
        logger.info(f"🔄 Creating raw data cache from {self.data_path}")
        data = self._load_raw_data()
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return data

    def _load_raw_data(self):
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
        return self._num_samples

    def __getitem__(self, idx):
        return {
            'source_ids': torch.from_numpy(self._src_ids[idx].astype(np.int64).copy()),
            'target_ids': torch.from_numpy(self._tgt_ids[idx].astype(np.int64).copy()),
            'source_mask': torch.from_numpy(self._src_mask[idx].copy()),
            'target_mask': torch.from_numpy(self._tgt_mask[idx].copy()),
            'vocab_pack_name': self._metadata[idx]['target_lang'],
            'vocab_size': self.config.model.vocab_size,
            'pad_token_id': 0,
            'unk_token_id': 1,
            'metadata': self._metadata[idx],
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ('_src_ids', '_tgt_ids', '_src_mask', '_tgt_mask'):
            state.pop(key, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._load_token_cache()

    def _pad_or_truncate(self, tokens, max_length):
        if len(tokens) > max_length:
            return tokens[:max_length]
        return tokens + [0] * (max_length - len(tokens))
