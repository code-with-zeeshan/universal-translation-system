# tools/create_vocabulary_packs.py
import json
import msgpack
import numpy as np
from collections import Counter
from transformers import AutoTokenizer
import sentencepiece as spm

class VocabularyPackCreator:
    def __init__(self, corpus_paths: Dict[str, str]):
        self.corpus_paths = corpus_paths
        self.tokenizer = spm.SentencePieceProcessor()
        
    def create_pack(self, languages: List[str], pack_name: str):
        """Create optimized vocabulary pack for language group"""
        
        # 1. Analyze corpus to find common tokens
        token_frequencies = self._analyze_corpus(languages)
        
        # 2. Select optimal vocabulary
        vocab = self._select_vocabulary(token_frequencies, target_size=25000)
        
        # 3. Create subword vocabulary for unknowns
        subwords = self._create_subword_vocab(vocab, token_frequencies)
        
        # 4. Optimize token IDs for compression
        optimized_vocab = self._optimize_token_ids(vocab, subwords)
        
        # 5. Create and save pack
        pack = {
            'name': pack_name,
            'version': '1.0',
            'languages': languages,
            'tokens': optimized_vocab['tokens'],
            'subwords': optimized_vocab['subwords'],
            'metadata': {
                'coverage': self._calculate_coverage(optimized_vocab, languages),
                'size_mb': len(msgpack.packb(optimized_vocab)) / 1024 / 1024
            }
        }
        
        # Save in multiple formats
        with open(f'{pack_name}.json', 'w') as f:
            json.dump(pack, f)
            
        with open(f'{pack_name}.msgpack', 'wb') as f:
            f.write(msgpack.packb(pack))
            
        return pack
    
    def _analyze_corpus(self, languages: List[str]) -> Counter:
        """Analyze corpus to find token frequencies"""
        token_freq = Counter()
        
        for lang in languages:
            corpus_path = self.corpus_paths.get(lang)
            if not corpus_path:
                continue
                
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    tokens = self.tokenizer.encode(line.strip())
                    token_freq.update(tokens)
                    
        return token_freq