# universal-decoder-node/universal_decoder_node/vocabulary.py
import msgpack
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass


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


class VocabularyManager:
    """Minimal vocabulary manager for decoder"""
    
    def __init__(self, vocab_dir: str = 'vocabs'):
        self.vocab_dir = Path(vocab_dir)
        self.loaded_packs = {}
        self.language_to_pack = self._build_language_mapping()
        
    def _build_language_mapping(self) -> Dict[str, str]:
        """Map languages to their vocabulary packs"""
        return {
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
    
    def get_vocab_for_pair(self, source_lang: str, target_lang: str) -> VocabularyPack:
        """Get appropriate vocabulary pack for language pair"""
        # Determine which pack to use
        source_pack = self.language_to_pack.get(source_lang, 'latin')
        target_pack = self.language_to_pack.get(target_lang, 'latin')
        
        # Use target language pack as priority
        pack_name = target_pack or source_pack or 'latin'
        
        # Load if not cached
        if pack_name not in self.loaded_packs:
            self.loaded_packs[pack_name] = self._load_pack(pack_name)
            
        return self.loaded_packs[pack_name]
    
    def _load_pack(self, pack_name: str) -> VocabularyPack:
        """Load vocabulary pack from disk"""
        # Try different versions
        for version in ['1.0', '1.1', '2.0']:
            pack_file = self.vocab_dir / f"{pack_name}_v{version}.msgpack"
            if pack_file.exists():
                break
        else:
            # Fallback to any version
            pack_files = list(self.vocab_dir.glob(f"{pack_name}_v*.msgpack"))
            if pack_files:
                pack_file = pack_files[0]
            else:
                raise FileNotFoundError(f"No vocabulary pack found for {pack_name}")
        
        with open(pack_file, 'rb') as f:
            data = msgpack.unpackb(f.read(), raw=False)
            
        # Handle different data formats
        if 'tokens' not in data:
            # Old format compatibility
            data = {
                'name': pack_name,
                'version': '1.0',
                'languages': [],
                'tokens': data.get('vocabulary', {}),
                'subwords': {},
                'special_tokens': data.get('special_tokens', {})
            }
            
        return VocabularyPack(**{
            'name': data.get('name', pack_name),
            'version': data.get('version', '1.0'),
            'languages': data.get('languages', []),
            'tokens': data.get('tokens', {}),
            'subwords': data.get('subwords', {}),
            'special_tokens': data.get('special_tokens', {})
        })