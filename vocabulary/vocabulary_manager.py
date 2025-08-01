# vocabulary/vocabulary_manager.py
import msgpack
import os
from utils.security import validate_path_component
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import asyncio
import aiofiles
import logging
from functools import lru_cache
import json
from utils.exceptions import VocabularyError

logger = logging.getLogger(__name__)

@dataclass
class VocabularyPack:
    name: str
    version: str
    languages: List[str]
    tokens: Dict[str, int]
    subwords: Dict[str, int]
    special_tokens: Dict[str, int]
    
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
        
        # Check ID range
        if all_ids:
            min_id, max_id = min(all_ids), max(all_ids)
            if min_id < 0:
                errors.append(f"Negative token ID found: {min_id}")
            expected_ids = set(range(max_id + 1))
            missing_ids = expected_ids - all_ids
            if len(missing_ids) > 100:  # Allow some gaps
                errors.append(f"Large gaps in token IDs: {len(missing_ids)} missing")
        
        return len(errors) == 0, errors

class VocabularyManager:
    def __init__(self, vocab_dir: str = 'vocabs', enable_async: bool = False):
        self.vocab_dir = Path(vocab_dir)
        self.loaded_packs = {}
        self.language_to_pack = self._build_language_mapping()
        self.enable_async = enable_async
        self._version_cache = {}

        # Validate vocabulary directory
        if not self.vocab_dir.exists():
            raise VocabularyError(f"Vocabulary directory not found: {vocab_dir}")
        
        # Load version information
        self._load_version_info()
        
        logger.info(f"VocabularyManager initialized with {len(self._version_cache)} packs")
        
    def _build_language_mapping(self) -> Dict[str, str]:
        """Map languages to their vocabulary packs with validation"""
        mapping = {
            # Latin languages
            'en': 'latin', 'es': 'latin', 'fr': 'latin', 
            'de': 'latin', 'it': 'latin', 'pt': 'latin',
            'nl': 'latin', 'sv': 'latin', 'pl': 'latin',
            'id': 'latin', 'vi': 'latin', 'tr': 'latin',
            # CJK
            'zh': 'cjk', 'ja': 'cjk', 'ko': 'cjk',
            # Others
            'ar': 'arabic', 'hi': 'devanagari',
            'ru': 'cyrillic', 'uk': 'cyrillic',
            'th': 'thai'
        }

        # Validate all mapped packs exist
        required_packs = set(mapping.values())
        for pack in required_packs:
            pack_files = list(self.vocab_dir.glob(f"{pack}_v*.msgpack"))
            if not pack_files:
                logger.warning(f"Vocabulary pack '{pack}' not found in {self.vocab_dir}")
        
        return mapping
    
    def _load_version_info(self):
        """Load version information for all packs"""
        self._version_cache = {}
        
        # Load from version manifest if exists
        manifest_path = self.vocab_dir / "manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r') as f:
                    self._version_cache = json.load(f)
                logger.info(f"Loaded version manifest with {len(self._version_cache)} entries")
            except Exception as e:
                logger.warning(f"Failed to load version manifest: {e}")
        
        # Scan directory for pack files
        for pack_file in self.vocab_dir.glob('*_v*.msgpack'):
            parts = pack_file.stem.split('_v')
            if len(parts) >= 2:
                pack_name = '_'.join(parts[:-1])
                version = parts[-1]
                
                if pack_name not in self._version_cache:
                    self._version_cache[pack_name] = []
                
                self._version_cache[pack_name].append({
                    'version': version,
                    'file': str(pack_file),
                    'size': pack_file.stat().st_size
                })
        
        # Sort versions for each pack
        for pack_name in self._version_cache:
            if isinstance(self._version_cache[pack_name], list):
                self._version_cache[pack_name].sort(
                    key=lambda x: [int(p) for p in x['version'].split('.')],
                    reverse=True
                )
    
    def get_vocab_for_pair(self, source_lang: str, target_lang: str, 
                          version: Optional[str] = None) -> VocabularyPack:
        """Get appropriate vocabulary pack for language pair with version support"""
        # Validate languages
        if source_lang not in self.language_to_pack:
            raise VocabularyError(f"Unsupported source language: {source_lang}")
        if target_lang not in self.language_to_pack:
            raise VocabularyError(f"Unsupported target language: {target_lang}")
        
        # Determine which pack to use
        source_pack = self.language_to_pack.get(source_lang)
        target_pack = self.language_to_pack.get(target_lang)
        
        # Use target language pack as priority
        pack_name = target_pack or source_pack or 'latin'
        
        # Create cache key with version
        cache_key = f"{pack_name}:{version}" if version else pack_name
        
        # Load if not cached
        if cache_key not in self.loaded_packs:
            self.loaded_packs[cache_key] = self._load_pack(pack_name, version)
            
        return self.loaded_packs[cache_key]
    
    def _load_pack(self, pack_name: str, version: Optional[str] = None) -> VocabularyPack:
        """Load vocabulary pack from disk with version support and security"""
        # Validate pack name to prevent path traversal
        pack_name = validate_path_component(pack_name)

        # Get available versions
        if pack_name not in self._version_cache:
            raise VocabularyError(f"No versions found for pack '{pack_name}'")
        
        available_versions = self._version_cache[pack_name]
        if not available_versions:
            raise VocabularyError(f"No versions found for pack '{pack_name}'")
        
        # Select version
        if version:
            # Find specific version
            version_info = next((v for v in available_versions if v['version'] == version), None)
            if not version_info:
                available = [v['version'] for v in available_versions]
                raise VocabularyError(f"Version {version} not found for {pack_name}. Available: {available}")
        else:
            # Use latest version
            version_info = available_versions[0]
        
        pack_file = Path(version_info['file'])
        logger.info(f"Loading {pack_name} v{version_info['version']} from {pack_file}")
        
        # Load with validation
        try:
            with open(pack_file, 'rb') as f:
                data = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
            
            # Create VocabularyPack
            pack = VocabularyPack(
                name=data.get('name', pack_name),
                version=data.get('version', version_info['version']),
                languages=data.get('languages', []),
                tokens=data.get('tokens', {}),
                subwords=data.get('subwords', {}),
                special_tokens=data.get('special_tokens', {})
            )
            
            # Validate pack
            is_valid, errors = pack.validate()
            if not is_valid:
                logger.warning(f"Vocabulary pack validation warnings: {errors}")
            
            return pack
            
        except Exception as e:
            logger.error(f"Failed to load vocabulary pack {pack_name}: {e}")
            raise

    def get_vocabulary_version_info(self) -> Dict[str, List[Dict[str, any]]]:
        """Get version info for all vocabulary packs"""
        return self._version_cache.copy()
    
    def get_loaded_versions(self) -> Dict[str, str]:
        """Get versions of currently loaded packs"""
        loaded = {}
        for cache_key, pack in self.loaded_packs.items():
            # Extract pack name from cache key
            pack_name = cache_key.split(':')[0]
            loaded[pack_name] = pack.version
        return loaded
    
    def get_latest_version(self, pack_name: str) -> Optional[str]:
        """Get latest version for a pack"""
        if pack_name in self._version_cache and self._version_cache[pack_name]:
            return self._version_cache[pack_name][0]['version']
        return None
    
    def preload_packs(self, languages: List[str], version: Optional[str] = None):
        """Preload vocabulary packs for given languages"""
        packs_to_load = set()
        
        for lang in languages:
            if lang in self.language_to_pack:
                packs_to_load.add(self.language_to_pack[lang])
        
        for pack_name in packs_to_load:
            try:
                self._load_pack(pack_name, version)
                logger.info(f"Preloaded {pack_name} for languages: {languages}")
            except Exception as e:
                logger.error(f"Failed to preload {pack_name}: {e}")
    
    async def get_vocab_for_pair_async(self, source_lang: str, target_lang: str,
                                      version: Optional[str] = None) -> VocabularyPack:
        """Async version of get_vocab_for_pair"""
        # Check cache first
        source_pack = self.language_to_pack.get(source_lang)
        target_pack = self.language_to_pack.get(target_lang)
        pack_name = target_pack or source_pack or 'latin'
        cache_key = f"{pack_name}:{version}" if version else pack_name
        
        if cache_key in self.loaded_packs:
            return self.loaded_packs[cache_key]
        
        # Load asynchronously
        return await self._load_pack_async(pack_name, version)
    
    async def _load_pack_async(self, pack_name: str, version: Optional[str] = None) -> VocabularyPack:
        """Async load vocabulary pack"""
        # Get version info
        if pack_name not in self._version_cache:
            raise VocabularyError(f"No versions found for pack '{pack_name}'")
        
        available_versions = self._version_cache[pack_name]
        version_info = available_versions[0] if not version else \
                      next((v for v in available_versions if v['version'] == version), None)
        
        if not version_info:
            raise VocabularyError(f"Version {version} not found for {pack_name}")
        
        pack_file = Path(version_info['file'])
        
        # Async file reading
        async with aiofiles.open(pack_file, 'rb') as f:
            content = await f.read()
        
        # Unpack in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None, 
            lambda: msgpack.unpackb(content, raw=False, strict_map_key=False)
        )
        
        # Create and cache pack
        pack = VocabularyPack(
            name=data.get('name', pack_name),
            version=data.get('version', version_info['version']),
            languages=data.get('languages', []),
            tokens=data.get('tokens', {}),
            subwords=data.get('subwords', {}),
            special_tokens=data.get('special_tokens', {})
        )
        
        cache_key = f"{pack_name}:{version}" if version else pack_name
        self.loaded_packs[cache_key] = pack
        
        return pack

    def get_vocabulary_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about loaded vocabularies"""
        stats = {
            'loaded_packs': len(self.loaded_packs),
            'total_memory_mb': 0,
            'pack_details': {},
            'language_coverage': {},
            'cache_hit_rate': 0
        }
    
        for pack_name, pack in self.loaded_packs.items():
            pack_stats = {
                'version': pack.version,
                'size': pack.size,
                'languages': pack.languages,
                'token_count': len(pack.tokens),
                'subword_count': len(pack.subwords),
                'special_token_count': len(pack.special_tokens)
            }
            stats['pack_details'][pack_name] = pack_stats
        
            # Calculate memory usage
            import sys
            stats['total_memory_mb'] += sys.getsizeof(pack) / (1024 * 1024)
        
            # Track language coverage
            for lang in pack.languages:
                stats['language_coverage'][lang] = stats['language_coverage'].get(lang, 0) + 1
    
        return stats
    
    def clear_cache(self):
        """Clear loaded vocabulary packs from memory"""
        self.loaded_packs.clear()
        logger.info("Cleared vocabulary cache")
    
    @lru_cache(maxsize=128)
    def get_token_id(self, token: str, pack_name: str) -> Optional[int]:
        """Cached token lookup"""
        pack = self._load_pack(pack_name)
        
        # Check in order: tokens, subwords, special_tokens
        if token in pack.tokens:
            return pack.tokens[token]
        elif token in pack.subwords:
            return pack.subwords[token]
        elif token in pack.special_tokens:
            return pack.special_tokens[token]
        
        return None

    # explicit cleanup
    def __del__(self):
        """Cleanup resources on deletion"""
        try:
            self.clear_cache()
            # Close any open file handles
            for pack_name, pack in self.loaded_packs.items():
                if hasattr(pack, '_file_handle') and pack._file_handle:
                    pack._file_handle.close()
            # Clear references
            self.loaded_packs.clear()
            self._version_cache.clear()
        except:
            pass  # Ignore errors during cleanup

    # Periodic cleanup method:
    def cleanup_old_cache(self, max_age_hours: int = 24):
        """Remove old cached packs not used recently"""
        current_time = time.time()
        to_remove = []
    
        for pack_name, pack in self.loaded_packs.items():
            if hasattr(pack, '_last_access_time'):
                age_hours = (current_time - pack._last_access_time) / 3600
                if age_hours > max_age_hours:
                    to_remove.append(pack_name)
    
        for pack_name in to_remove:
            del self.loaded_packs[pack_name]
            logger.info(f"Removed old cached pack: {pack_name}")

class VocabularyMigrator:
    """Handle vocabulary pack version migrations"""
    
    @staticmethod
    def migrate_pack(old_pack: VocabularyPack, target_version: str) -> VocabularyPack:
        """Migrate vocabulary pack to target version"""
        migrations = {
            ('1.0', '2.0'): VocabularyMigrator._migrate_1_0_to_2_0,
            ('2.0', '3.0'): VocabularyMigrator._migrate_2_0_to_3_0,
        }
        
        current_version = old_pack.version
        migration_key = (current_version, target_version)
        
        if migration_key in migrations:
            return migrations[migration_key](old_pack)
        else:
            raise VocabularyError(f"No migration path from {current_version} to {target_version}")
    
    @staticmethod
    def _migrate_1_0_to_2_0(old_pack: VocabularyPack) -> VocabularyPack:
        """Migrate from version 1.0 to 2.0"""
        # Add new required fields, convert formats, etc.
        new_pack = VocabularyPack(
            name=old_pack.name,
            version='2.0',
            languages=old_pack.languages,
            tokens=old_pack.tokens,
            subwords=old_pack.subwords,
            special_tokens={
                **old_pack.special_tokens,
                '<mask>': len(old_pack.tokens) + len(old_pack.subwords) + len(old_pack.special_tokens)
            }
        )
        return new_pack   

class VocabularyAnalytics:
    """Track vocabulary usage patterns"""
    
    def __init__(self):
        self.token_usage = {}
        self.unknown_tokens = []
        self.language_pair_usage = {}
    
    def record_tokenization(self, text: str, tokens: List[int], language_pair: str):
        """Record tokenization event for analytics"""
        self.language_pair_usage[language_pair] = self.language_pair_usage.get(language_pair, 0) + 1
        
        for token_id in tokens:
            self.token_usage[token_id] = self.token_usage.get(token_id, 0) + 1
    
    def get_usage_report(self) -> Dict[str, Any]:
        """Generate usage report"""
        return {
            'total_tokenizations': sum(self.language_pair_usage.values()),
            'unique_tokens_used': len(self.token_usage),
            'most_used_tokens': sorted(self.token_usage.items(), key=lambda x: x[1], reverse=True)[:100],
            'language_pair_distribution': self.language_pair_usage,
            'unknown_token_samples': self.unknown_tokens[-100:]
        } 

    def health_check(self) -> Dict[str, bool]:
        """Perform health check on vocabulary system"""
        checks = {
            'vocab_dir_exists': self.vocab_dir.exists(),
            'vocab_dir_readable': os.access(self.vocab_dir, os.R_OK),
            'manifest_exists': (self.vocab_dir / "manifest.json").exists(),
            'has_vocabulary_packs': len(list(self.vocab_dir.glob('*_v*.msgpack'))) > 0,
            'cache_operational': True,
            'all_languages_mapped': all(lang in self.language_to_pack for lang in ['en', 'es', 'fr', 'de', 'zh'])
        }
    
        # Test cache operations
        try:
            test_pack = self._load_pack('latin')
            if not test_pack:
                checks['cache_operational'] = False
        except:
            checks['cache_operational'] = False
    
        return checks               

# Utility functions
def create_vocabulary_manifest(vocab_dir: str = 'vocabs'):
    """Create manifest file for vocabulary packs"""
    vocab_path = Path(vocab_dir)
    manifest = {}
    
    for pack_file in vocab_path.glob('*_v*.msgpack'):
        parts = pack_file.stem.split('_v')
        if len(parts) >= 2:
            pack_name = '_'.join(parts[:-1])
            version = parts[-1]
            
            if pack_name not in manifest:
                manifest[pack_name] = []
            
            # Load pack to get metadata
            try:
                with open(pack_file, 'rb') as f:
                    data = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
                
                manifest[pack_name].append({
                    'version': version,
                    'file': str(pack_file),
                    'size': pack_file.stat().st_size,
                    'languages': data.get('languages', []),
                    'total_tokens': len(data.get('tokens', {})) + 
                                   len(data.get('subwords', {})) + 
                                   len(data.get('special_tokens', {}))
                })
            except Exception as e:
                logger.error(f"Failed to read {pack_file}: {e}")
    
    # Sort versions
    for pack_name in manifest:
        manifest[pack_name].sort(
            key=lambda x: [int(p) for p in x['version'].split('.')],
            reverse=True
        )
    
    # Save manifest
    manifest_path = vocab_path / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Created vocabulary manifest with {len(manifest)} packs")
    return manifest