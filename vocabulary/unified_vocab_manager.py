# vocabulary/unified_vocab_manager.py
"""
Unified vocabulary manager for the Universal Translation System.
This module provides a unified interface for vocabulary management.
"""
import mmap
import msgpack
# Optional compression dependency
try:
    import zstandard as zstd  # type: ignore
except Exception:  # pragma: no cover
    zstd = None  # type: ignore
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any, Union, Set
import threading
import asyncio
import aiofiles
import logging
from enum import Enum
from dataclasses import dataclass
import os
import json

from utils.exceptions import VocabularyError
from utils.security import validate_path_component
from utils.base_classes import BaseVocabularyManager, TokenizerMixin
from utils.thread_safety import THREAD_SAFETY_INTERNAL, thread_safe, document_thread_safety as _document_thread_safety
from utils.constants import (
    VOCAB_SIZE, VOCAB_MIN_FREQUENCY, VOCAB_SPECIAL_TOKENS,
    VOCAB_PAD_ID, VOCAB_UNK_ID, VOCAB_BOS_ID, VOCAB_EOS_ID
)
from config.schemas import RootConfig

class VocabularyMode(Enum):
    """Operating modes for vocabulary manager"""
    FULL = "full"          # Full features, high memory
    OPTIMIZED = "optimized"  # Balanced memory/features
    EDGE = "edge"          # Minimal memory for edge devices

@dataclass
class VocabularyPack:
    """Unified vocabulary pack supporting different modes"""
    name: str
    version: str
    languages: List[str]
    tokens: Dict[str, int]
    subwords: Dict[str, int]
    special_tokens: Dict[str, int]
    mode: VocabularyMode = VocabularyMode.FULL
    
    # Optional optimizations
    bloom_filter: Optional[Any] = None
    prefix_tree: Optional[Dict] = None
    compressed_data: Optional[bytes] = None

    def __post_init__(self):
        """Initialize derived data structures"""
        self.id_to_token = {v: k for k, v in self.tokens.items()}
        if self.subwords:
            self.id_to_subword = {v: k for k, v in self.subwords.items()}
        
    def contains_token(self, token: str) -> bool:
        """Check if token exists in vocabulary"""
        if self.bloom_filter:
            if token not in self.bloom_filter:
                return False
        return token in self.tokens or token in self.subwords
    
    @property
    def size(self) -> int:
        return len(self.tokens) + len(self.subwords) + len(self.special_tokens)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate vocabulary pack integrity"""
        errors = []
        
        # Check required special tokens
        required_special = ['<pad>', '<unk>', '<s>', '</s>']
        for token in required_special:
            if token not in self.special_tokens:
                errors.append(f"Missing required special token: {token}")
        
        # Check for ID conflicts
        all_ids = set()
        for token_dict in [self.tokens, self.subwords, self.special_tokens]:
            for token, id_ in token_dict.items():
                if id_ in all_ids:
                    errors.append(f"Duplicate ID {id_} for token {token}")
                all_ids.add(id_)
        
        return len(errors) == 0, errors
    
    def optimize_for_edge(self, max_tokens: int = 8000):
        """Convert to edge-optimized version"""
        if self.mode == VocabularyMode.EDGE:
            return
        
        # Filter tokens
        self.tokens = dict(list(self.tokens.items())[:max_tokens])
        
        # Build optimization structures
        self._build_bloom_filter()
        self._build_prefix_tree()
        
        # Compress if needed
        if self.mode == VocabularyMode.EDGE:
            self._compress_data()
        
        self.mode = VocabularyMode.EDGE
    
    def _build_bloom_filter(self):
        """Build bloom filter for fast negative lookups"""
        try:
            from pybloom_live import BloomFilter
            self.bloom_filter = BloomFilter(
                capacity=len(self.tokens) * 2, 
                error_rate=0.001
            )
            for token in self.tokens:
                self.bloom_filter.add(token)
        except ImportError:
            self.bloom_filter = None
    
    def _build_prefix_tree(self):
        """Build prefix tree for efficient subword tokenization"""
        self.prefix_tree = {}
        for token in self.tokens:
            node = self.prefix_tree
            for char in token:
                if char not in node:
                    node[char] = {}
                node = node[char]
            node['$'] = self.tokens[token]
    
    def _compress_data(self):
        """Compress vocabulary data for minimal memory. If zstd is unavailable, skip compression."""
        if zstd is None:  # graceful degrade in smoke/dry-run
            return
        compressor = zstd.ZstdCompressor(level=3)
        self.compressed_data = compressor.compress(
            msgpack.packb({
                'tokens': self.tokens,
                'subwords': self.subwords
            })
        )
    
    def has_token(self, token: str) -> bool:
        """Fast token existence check"""
        if self.bloom_filter:
            if token not in self.bloom_filter:
                return False
        return token in self.tokens or token in self.subwords

class UnifiedVocabularyManager(BaseVocabularyManager, TokenizerMixin):
    """
    Unified vocabulary manager combining features from both implementations.
    Supports multiple operating modes for different deployment scenarios.

    Thread Safety:
        All public methods are thread-safe with internal synchronization.
    """
    
    def __init__(self, 
                 config: RootConfig,
                 vocab_dir: str = 'vocabs',
                 mode: VocabularyMode = VocabularyMode.OPTIMIZED,
                 cache_size: Optional[int] = None,
                 enable_async: bool = False):
        """
        Initialize unified vocabulary manager.
        
        Args:
            config: Root configuration
            vocab_dir: Directory containing vocabulary packs
            mode: Operating mode (FULL, OPTIMIZED, or EDGE)
            cache_size: Maximum cached packs (None for unlimited in FULL mode)
            enable_async: Enable async operations
        """
        super().__init__(config, vocab_dir)
        
        self.mode = mode
        self.enable_async = enable_async
        
        # Set cache size based on mode
        if cache_size is None:
            self.cache_size = {
                VocabularyMode.FULL: None,  # Unlimited
                VocabularyMode.OPTIMIZED: 10,
                VocabularyMode.EDGE: 3
            }[mode]
        else:
            self.cache_size = cache_size
        
        # Thread safety
        self._lock = threading.Lock()
        
        # LRU cache for loaded packs
        self._vocabulary_cache = {}
        self._cache_order = []
        
        # Version information
        self._version_cache = {}
        
        # Analytics (only in FULL mode)
        if self.mode == VocabularyMode.FULL:
            from collections import Counter
            self.token_usage = Counter()
            self.unknown_token_usage = Counter()
            self.language_pair_usage = Counter()
        
        # Initialize
        self._load_metadata()
        
        self.logger.info(
            f"UnifiedVocabularyManager initialized in {mode.value} mode "
            f"with cache_size={self.cache_size}"
        )

    def _load_metadata(self):
        """Load metadata for all packs (lightweight)"""
        # Load version manifest if exists
        manifest_path = self.vocab_dir / "manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r') as f:
                    self._version_cache = json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load manifest: {e}")
        
        # Scan for pack files
        self.pack_metadata = {}
        for pack_file in self.vocab_dir.glob('*_v*.msgpack'):
            pack_name = pack_file.stem.split('_v')[0]
            self.pack_metadata[pack_name] = {
                'file': pack_file,
                'size': pack_file.stat().st_size
            }
    
    def get_vocab_for_pair(self, 
                          source_lang: str, 
                          target_lang: str,
                          version: Optional[str] = None) -> VocabularyPack:
        """
        Get vocabulary pack for language pair.
        
        This is the main interface method that adapts based on mode.
        """
        # Validate languages
        if source_lang not in self.language_to_pack:
            raise VocabularyError(f"Unsupported source language: {source_lang}")
        if target_lang not in self.language_to_pack:
            raise VocabularyError(f"Unsupported target language: {target_lang}")
        
        # Determine pack names
        source_pack = self.language_to_pack[source_lang]
        target_pack = self.language_to_pack[target_lang]
        
        # Handle cross-language pairs in EDGE mode
        if self.mode == VocabularyMode.EDGE and source_pack != target_pack:
            return self._create_mini_pack(source_lang, target_lang, source_pack, target_pack)
        
        # Use target pack as priority
        pack_name = target_pack or source_pack or 'latin'
        
        # Load with caching
        return self._load_pack_cached(pack_name, version)
    
    def _load_pack_cached(self, pack_name: str, version: Optional[str] = None) -> VocabularyPack:
        """Load pack with LRU caching"""
        cache_key = f"{pack_name}:{version}" if version else pack_name
        
        with self._lock:
            # Check cache
            if cache_key in self._vocabulary_cache:
                # Move to end (most recently used)
                if self.cache_size:  # Only track order if cache is limited
                    self._cache_order.remove(cache_key)
                    self._cache_order.append(cache_key)
                return self._vocabulary_cache[cache_key]
            
            # Load from file
            pack = self._load_pack_from_file(pack_name, version)
            
            # Optimize based on mode
            if self.mode == VocabularyMode.EDGE:
                pack.optimize_for_edge()
            elif self.mode == VocabularyMode.OPTIMIZED:
                pack.optimize_for_edge(max_tokens=16000)  # Keep more tokens
            
            # Add to cache
            self._vocabulary_cache[cache_key] = pack
            
            # Handle cache eviction if size limited
            if self.cache_size:
                self._cache_order.append(cache_key)
                
                if len(self._cache_order) > self.cache_size:
                    evict_key = self._cache_order.pop(0)
                    del self._vocabulary_cache[evict_key]
                    self.logger.debug(f"Evicted {evict_key} from cache")
            
            return pack
    
    def _load_pack_from_file(self, pack_name: str, version: Optional[str] = None) -> VocabularyPack:
        """Load vocabulary pack from disk with security validation"""
        # Validate pack name
        pack_name = validate_path_component(pack_name)
        
        # Find pack file
        if pack_name not in self.pack_metadata:
            raise VocabularyError(f"Pack not found: {pack_name}")
        
        pack_file = self.pack_metadata[pack_name]['file']
        
        try:
            # Load data with safety limits
            with open(pack_file, 'rb') as f:
                data = self._safe_unpack_msgpack(f.read())
            
            # Create vocabulary pack
            pack = VocabularyPack(
                name=data.get('name', pack_name),
                version=data.get('version', '1.0'),
                languages=data.get('languages', []),
                tokens=data.get('tokens', {}),
                subwords=data.get('subwords', {}),
                special_tokens=data.get('special_tokens', {}),
                mode=self.mode
            )
            
            # Validate
            is_valid, errors = pack.validate()
            if not is_valid and self.mode == VocabularyMode.FULL:
                self.logger.warning(f"Validation warnings for {pack_name}: {errors}")
            
            return pack
            
        except Exception as e:
            self.logger.error(f"Failed to load pack {pack_name}: {e}")
            raise
    
    def _safe_unpack_msgpack(self, data: bytes, max_size: int = 50 * 1024 * 1024) -> dict:
        """Safely unpack MessagePack data with validation"""
        if len(data) > max_size:
            raise VocabularyError(f"Data too large: {len(data)} bytes")
        
        try:
            unpacked = msgpack.unpackb(
                data,
                raw=False,
                strict_map_key=False,
                max_str_len=1024 * 1024,
                max_bin_len=10 * 1024 * 1024,
                max_array_len=1000000,
                max_map_len=1000000
            )
            
            if not isinstance(unpacked, dict):
                raise VocabularyError(f"Expected dict, got {type(unpacked)}")
            
            return unpacked
            
        except msgpack.exceptions.UnpackException as e:
            raise VocabularyError(f"Invalid MessagePack data: {e}")
    
    def _create_mini_pack(self, 
                         source_lang: str, 
                         target_lang: str,
                         source_pack_name: str,
                         target_pack_name: str) -> VocabularyPack:
        """Create minimal merged pack for cross-language pairs (EDGE mode)"""
        # Load both packs
        source_pack = self._load_pack_cached(source_pack_name)
        target_pack = self._load_pack_cached(target_pack_name)
        
        # Merge tokens (limited for edge)
        merged_tokens = {}
        max_per_pack = 5000 if self.mode == VocabularyMode.EDGE else 8000
        
        # Take top tokens from each
        for token, idx in list(source_pack.tokens.items())[:max_per_pack]:
            merged_tokens[token] = idx
        
        for token, idx in list(target_pack.tokens.items())[:max_per_pack]:
            if token not in merged_tokens:
                merged_tokens[token] = len(merged_tokens) + idx
        
        # Combine special tokens
        merged_special = {**source_pack.special_tokens, **target_pack.special_tokens}
        
        # Create mini pack
        mini_pack = VocabularyPack(
            name=f"{source_lang}-{target_lang}",
            version="merged",
            languages=[source_lang, target_lang],
            tokens=merged_tokens,
            subwords={},  # Skip subwords for mini packs
            special_tokens=merged_special,
            mode=self.mode
        )
        
        if self.mode == VocabularyMode.EDGE:
            mini_pack.optimize_for_edge()
        
        return mini_pack
    
    def tokenize(self, text: str, language: str, pack: Optional[VocabularyPack] = None) -> List[int]:
        """
        Unified tokenization method.
        Uses TokenizerMixin for consistency or pack's optimized method.
        """
        if pack is None:
            # Auto-load pack for language
            lang_pack_name = self.language_to_pack.get(language, 'latin')
            pack = self._load_pack_cached(lang_pack_name)
        
        # Use optimized tokenization for EDGE mode
        if self.mode == VocabularyMode.EDGE and pack.prefix_tree:
            return self._tokenize_with_prefix_tree(text, language, pack)
        
        # Use standard tokenization from mixin
        return self.tokenize_with_subwords(text, pack, language)
    
    def _tokenize_with_prefix_tree(self, text: str, language: str, pack: VocabularyPack) -> List[int]:
        """Optimized tokenization using prefix tree (EDGE mode)"""
        tokens = [pack.special_tokens.get(f'<{language}>', pack.special_tokens['<s>'])]
        
        for word in text.lower().split():
            if word in pack.tokens:
                tokens.append(pack.tokens[word])
            else:
                # Use prefix tree for subword tokenization
                subwords = self._subword_with_prefix_tree(word, pack)
                tokens.extend(subwords)
        
        tokens.append(pack.special_tokens['</s>'])
        return tokens
    
    def _subword_with_prefix_tree(self, word: str, pack: VocabularyPack) -> List[int]:
        """Subword tokenization using prefix tree"""
        if not pack.prefix_tree:
            return [pack.special_tokens['<unk>']]
        
        subwords = []
        i = 0
        
        while i < len(word):
            node = pack.prefix_tree
            longest_match = None
            j = i
            
            while j < len(word) and word[j] in node:
                node = node[word[j]]
                if '$' in node:
                    longest_match = (j + 1, node['$'])
                j += 1
            
            if longest_match:
                subwords.append(longest_match[1])
                i = longest_match[0]
            else:
                subwords.append(pack.special_tokens['<unk>'])
                i += 1
        
        return subwords if subwords else [pack.special_tokens['<unk>']]
    
    def record_usage(self, text: str, tokens: List[int], language_pair: str, pack: VocabularyPack):
        """Record usage analytics (FULL mode only)"""
        if self.mode != VocabularyMode.FULL:
            return
        
        self.language_pair_usage[language_pair] += 1
        self.token_usage.update(tokens)
        
        # Track unknown tokens
        unk_id = pack.special_tokens.get('<unk>', 1)
        for word in text.lower().split():
            if word not in pack.tokens and unk_id in tokens:
                self.unknown_token_usage[word] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        stats = {
            'mode': self.mode.value,
            'loaded_packs': len(self._vocabulary_cache),
            'cache_size': self.cache_size,
            'available_packs': len(self.pack_metadata)
        }
        
        if self.mode == VocabularyMode.FULL:
            stats.update({
                'total_tokenizations': sum(self.language_pair_usage.values()),
                'unique_tokens_used': len(self.token_usage),
                'most_common_unknowns': self.unknown_token_usage.most_common(10),
                'language_pairs': dict(self.language_pair_usage)
            })
        
        # Memory usage estimate
        import sys
        total_memory = sum(
            sys.getsizeof(pack) for pack in self._vocabulary_cache.values()
        )
        stats['memory_mb'] = total_memory / (1024 * 1024)
        
        return stats
    
    def preload_for_languages(self, languages: List[str]):
        """Preload vocabulary packs for specified languages"""
        packs_to_load = set()
        
        for lang in languages:
            if lang in self.language_to_pack:
                packs_to_load.add(self.language_to_pack[lang])
        
        for pack_name in packs_to_load:
            try:
                self._load_pack_cached(pack_name)
                self.logger.info(f"Preloaded {pack_name}")
            except Exception as e:
                self.logger.error(f"Failed to preload {pack_name}: {e}")
    
    async def get_vocab_for_pair_async(self, 
                                       source_lang: str,
                                       target_lang: str,
                                       version: Optional[str] = None) -> VocabularyPack:
        """Async version for non-blocking operations"""
        if not self.enable_async:
            raise RuntimeError("Async not enabled. Initialize with enable_async=True")
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.get_vocab_for_pair,
            source_lang,
            target_lang,
            version
        )
    
    def clear_cache(self):
        """Clear vocabulary cache to free memory"""
        with self._lock:
            self._vocabulary_cache.clear()
            self._cache_order.clear()
            self.logger.info("Vocabulary cache cleared")
    
    def health_check(self) -> Dict[str, bool]:
        """System health check"""
        checks = {
            'vocab_dir_exists': self.vocab_dir.exists(),
            'vocab_dir_readable': os.access(self.vocab_dir, os.R_OK),
            'has_vocabulary_packs': len(self.pack_metadata) > 0,
            'cache_operational': True,
            'mode_configured': self.mode in VocabularyMode
        }
        
        # Test cache operation
        try:
            if self.pack_metadata:
                test_pack = list(self.pack_metadata.keys())[0]
                self._load_pack_cached(test_pack)
        except:
            checks['cache_operational'] = False
        
        return checks