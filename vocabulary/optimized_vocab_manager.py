# vocabulary/optimized_vocab_manager.py
import mmap
import msgpack
from functools import lru_cache
from pathlib import Path
import threading
from typing import Dict, Optional, Tuple, List
import numpy as np
from utils.common_utils import StandardLogger
from utils.exceptions import VocabularyError
import zstandard as zstd

class OptimizedVocabularyManager:
    """Memory-efficient vocabulary management for edge devices"""
    
    def __init__(self, vocab_dir: str = 'vocabs', cache_size: int = 3):
        self.vocab_dir = Path(vocab_dir)
        self.cache_size = cache_size
        self._lock = threading.Lock()
        self.logger = StandardLogger.get_logger(__name__)
        
        # Memory-mapped files
        self.mmap_files = {}
        
        # LRU cache for hot vocabularies
        self._vocabulary_cache = {}
        self._cache_order = []
        
        # Language to pack mapping
        self.language_to_pack = {
            'en': 'latin', 'es': 'latin', 'fr': 'latin', 'de': 'latin',
            'it': 'latin', 'pt': 'latin', 'nl': 'latin', 'sv': 'latin',
            'zh': 'cjk', 'ja': 'cjk', 'ko': 'cjk',
            'ar': 'arabic', 'hi': 'devanagari',
            'ru': 'cyrillic', 'uk': 'cyrillic',
            'th': 'thai', 'vi': 'latin', 'pl': 'latin',
            'tr': 'latin', 'id': 'latin'
        }
        
        # Preload metadata only
        self._load_pack_metadata()
    
    def _load_pack_metadata(self):
        """Load only metadata for all packs (very lightweight)"""
        self.pack_metadata = {}
        
        for pack_file in self.vocab_dir.glob('*_v*.msgpack'):
            pack_name = pack_file.stem.split('_v')[0]
            
            # Read just the header (first 1KB)
            with open(pack_file, 'rb') as f:
                header_data = f.read(1024)
                # Simple parsing - in production use proper msgpack streaming
                if b'"languages"' in header_data:
                    # Extract language list from header
                    self.pack_metadata[pack_name] = {
                        'file': pack_file,
                        'size': pack_file.stat().st_size
                    }
    
    def get_vocabulary_for_edge(self, source_lang: str, target_lang: str) -> 'EdgeVocabularyPack':
        """Get minimal vocabulary for edge device"""
        # Determine pack
        source_pack = self.language_to_pack.get(source_lang, 'latin')
        target_pack = self.language_to_pack.get(target_lang, 'latin')
        
        # If different packs needed, create merged mini-pack
        if source_pack != target_pack:
            return self._create_mini_pack(source_lang, target_lang)
        
        # Load from cache or file
        return self._load_pack_cached(source_pack)
    
    def _load_pack_cached(self, pack_name: str) -> 'EdgeVocabularyPack':
        """Load pack with LRU caching"""
        with self._lock:
            # Check cache
            if pack_name in self._vocabulary_cache:
                # Move to end (most recently used)
                self._cache_order.remove(pack_name)
                self._cache_order.append(pack_name)
                return self._vocabulary_cache[pack_name]
            
            # Load from file
            pack = self._load_pack_from_file(pack_name)
            
            # Add to cache
            self._vocabulary_cache[pack_name] = pack
            self._cache_order.append(pack_name)
            
            # Evict if cache full
            if len(self._cache_order) > self.cache_size:
                evict_pack = self._cache_order.pop(0)
                del self._vocabulary_cache[evict_pack]
                self.logger.info(f"Evicted {evict_pack} from cache")
            
            return pack

    def _safe_unpack_msgpack(self, data: bytes, max_size: int = 50 * 1024 * 1024) -> dict:
        """
        Safely unpack MessagePack data with validation.
    
        Args:
            data: MessagePack bytes to unpack
            max_size: Maximum allowed size in bytes (default 50MB)
        
        Returns:
            Unpacked dictionary
        
        Raises:
            VocabularyError: If data is invalid or too large
        """
        import msgpack
    
        # Check data size
        if len(data) > max_size:
            raise VocabularyError(f"MessagePack data too large: {len(data)} bytes (max: {max_size})")
    
        try:
            # Unpack with safety limits
            unpacked = msgpack.unpackb(
               data, 
               raw=False,
               strict_map_key=False,  # Allow non-string keys for compatibility
               max_str_len=1024 * 1024,      # 1MB max string length
               max_bin_len=10 * 1024 * 1024,  # 10MB max binary length
               max_array_len=1000000,          # 1M max array length
               max_map_len=1000000,            # 1M max map length
               max_ext_len=1024 * 1024         # 1MB max extension type
            )
        
            # Validate structure
            if not isinstance(unpacked, dict):
                raise VocabularyError(f"Expected dict, got {type(unpacked)}")
        
            # Validate required fields
            required_fields = ['name', 'tokens', 'special_tokens']
            missing_fields = [field for field in required_fields if field not in unpacked]
            if missing_fields:
                raise VocabularyError(f"Missing required fields: {missing_fields}")
        
            # Validate data types
            if not isinstance(unpacked.get('tokens', {}), dict):
                raise VocabularyError("'tokens' must be a dictionary")
        
            if not isinstance(unpacked.get('special_tokens', {}), dict):
                raise VocabularyError("'special_tokens' must be a dictionary")
        
            return unpacked
        
        except msgpack.exceptions.UnpackException as e:
            raise VocabularyError(f"Invalid MessagePack data: {e}")
        except Exception as e:
            logger.error(f"Failed to unpack MessagePack data: {e}")
            raise
    
    def _load_pack_from_file(self, pack_name: str) -> 'EdgeVocabularyPack':
        """Load vocabulary pack optimized for edge with validation"""
        pack_file = self.pack_metadata[pack_name]['file']
        
        # For edge devices, load only essential tokens
        with open(pack_file, 'rb') as f:
            # Use safe unpacking
            full_data = self._safe_unpack_msgpack(f.read())
        
        # Create edge-optimized version
        edge_pack = EdgeVocabularyPack(
            name=pack_name,
            tokens=self._filter_essential_tokens(full_data['tokens']),
            special_tokens=full_data['special_tokens']
        )
        
        return edge_pack
    
    def _filter_essential_tokens(self, tokens: Dict[str, int], max_tokens: int = 8000) -> Dict[str, int]:
        """Keep only most frequent tokens for edge"""
        # In production, sort by frequency
        # For now, keep first N tokens
        essential = {}
        for token, idx in list(tokens.items())[:max_tokens]:
            essential[token] = idx
        return essential
    
    def _create_mini_pack(self, source_lang: str, target_lang: str) -> 'EdgeVocabularyPack':
        """Create minimal pack for specific language pair"""
        # Load both packs
        source_pack = self._load_pack_cached(self.language_to_pack[source_lang])
        target_pack = self._load_pack_cached(self.language_to_pack[target_lang])
        
        # Merge essential tokens only
        merged_tokens = {}
        merged_tokens.update(source_pack.tokens)
        merged_tokens.update(target_pack.tokens)
        
        # Keep only top 10K tokens for cross-language pairs
        if len(merged_tokens) > 10000:
            merged_tokens = dict(list(merged_tokens.items())[:10000])
        
        return EdgeVocabularyPack(
            name=f"{source_lang}-{target_lang}",
            tokens=merged_tokens,
            special_tokens={**source_pack.special_tokens, **target_pack.special_tokens}
        )
    
    def preload_user_languages(self, user_languages: List[str]):
        """Preload vocabularies based on user's language preferences"""
        # Determine which packs needed
        needed_packs = set()
        for lang in user_languages:
            pack = self.language_to_pack.get(lang, 'latin')
            needed_packs.add(pack)
        
        # Preload in background
        for pack in needed_packs:
            self._load_pack_cached(pack)
        
        self.logger.info(f"Preloaded packs for languages: {user_languages}")

class EdgeVocabularyPack:
    """Lightweight vocabulary pack for edge devices"""
    
    def __init__(self, name: str, tokens: Dict[str, int], special_tokens: Dict[str, int]):
        self.name = name
        self.tokens = tokens
        self.special_tokens = special_tokens
        self.all_tokens = {**special_tokens, **tokens}
        
        # Create reverse mapping for decoding
        self.id_to_token = {v: k for k, v in self.all_tokens.items()}
        
        # Quick lookup structures
        self._build_lookup_structures()

        # Add bloom filter for O(1) negative lookups
        self._init_bloom_filter()

    def _init_bloom_filter(self):
        """Initialize bloom filter for fast token existence checks"""
        try:
            from pybloom_live import BloomFilter
            self.bloom = BloomFilter(capacity=len(self.all_tokens) * 2, error_rate=0.001)
            for token in self.all_tokens:
                self.bloom.add(token)
        except ImportError:
            self.bloom = None  # Fallback to regular lookup
    
    def has_token(self, token: str) -> bool:
        """Fast token existence check"""
        if self.bloom:
            if token not in self.bloom:
                return False  # Definitely not present
        return token in self.all_tokens
    
    def _build_lookup_structures(self):
        """Build efficient lookup structures"""
        # Prefix tree for subword tokenization
        self.prefix_tree = {}
        for token in self.tokens:
            node = self.prefix_tree
            for char in token:
                if char not in node:
                    node[char] = {}
                node = node[char]
            node['$'] = self.tokens[token]  # End marker with token ID
    
    def tokenize(self, text: str, language: str) -> List[int]:
        """Fast tokenization for edge devices"""
        # Add language token
        tokens = [self.special_tokens.get(f'<{language}>', self.special_tokens['<unk>'])]
        
        # Simple whitespace tokenization with subword fallback
        for word in text.lower().split():
            if word in self.tokens:
                tokens.append(self.tokens[word])
            else:
                # Subword tokenization using prefix tree
                subwords = self._subword_tokenize(word)
                tokens.extend(subwords)
        
        # Add end token
        tokens.append(self.special_tokens['</s>'])
        
        return tokens
    
    def _subword_tokenize(self, word: str) -> List[int]:
        """Efficient subword tokenization using prefix tree"""
        subwords = []
        i = 0
        
        while i < len(word):
            # Find longest matching prefix
            node = self.prefix_tree
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
                # Unknown character, use UNK
                subwords.append(self.special_tokens['<unk>'])
                i += 1
        
        return subwords if subwords else [self.special_tokens['<unk>']]

class CompressedVocabularyPack:
    """Compressed vocabulary pack for ultra-low memory devices"""
    
    def __init__(self, pack: EdgeVocabularyPack):
        # Compress token dictionary
        self.compressed_tokens = zstd.compress(
            msgpack.packb(pack.tokens),
            level=3
        )
        self.special_tokens = pack.special_tokens  # Keep these uncompressed
        self._decompressor = zstd.ZstdDecompressor()
        self._tokens_cache = None
    
    @property
    def tokens(self):
        if self._tokens_cache is None:
            compressed_data = self._decompressor.decompress(self.compressed_tokens)
            self._tokens_cache = msgpack.unpackb(compressed_data)
        return self._tokens_cache