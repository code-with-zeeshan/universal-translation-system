# vocabulary/create_vocabulary_packs_from_data.py
import sentencepiece as spm
from collections import Counter
import json
import msgpack

class VocabularyPackCreator:
    def __init__(self, corpus_dir='data/processed'):
        self.corpus_dir = corpus_dir
        
    def create_all_packs(self):
        """Create all vocabulary packs"""
        
        # Language groupings
        groups = {
            'latin': ['en', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'sv', 'pl', 'id', 'vi', 'tr'],
            'cjk': ['zh', 'ja', 'ko'],
            'arabic': ['ar'],
            'devanagari': ['hi'],
            'cyrillic': ['ru', 'uk'],
            'thai': ['th']
        }
        
        for group_name, languages in groups.items():
            print(f"\n📦 Creating {group_name} vocabulary pack...")
            self.create_pack(group_name, languages)
    
    def create_pack(self, pack_name, languages):
        """Create vocabulary pack for language group"""
        
        # 1. Merge corpora for all languages in group
        merged_corpus = f"temp_{pack_name}_corpus.txt"
        self._merge_corpora(languages, merged_corpus)
        
        # 2. Train SentencePiece model
        spm.SentencePieceTrainer.train(
            input=merged_corpus,
            model_prefix=f'vocabs/{pack_name}',
            vocab_size=25000,
            model_type='bpe',
            character_coverage=0.9995,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            split_by_whitespace=True,
            num_threads=16
        )
        
        # 3. Load trained model and create pack
        sp = spm.SentencePieceProcessor()
        sp.load(f'vocabs/{pack_name}.model')
        
        # 4. Create vocabulary mappings
        vocab = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
        subwords = {}
        
        for i in range(sp.get_piece_size()):
            piece = sp.id_to_piece(i)
            if piece.startswith('▁'):  # SentencePiece word boundary
                vocab[piece[1:]] = i
            elif piece.startswith('##'):
                subwords[piece] = i
            else:
                vocab[piece] = i
        
        # 5. Add language tokens
        for lang in languages:
            vocab[f'<{lang}>'] = len(vocab)
        
        # 6. Create and save pack
        pack = {
            'name': pack_name,
            'version': '1.0',
            'languages': languages,
            'tokens': vocab,
            'subwords': subwords,
            'size_mb': len(msgpack.packb(vocab)) / 1024 / 1024
        }
        
        # Save in multiple formats
        with open(f'vocabs/{pack_name}_v1.json', 'w', encoding='utf-8') as f:
            json.dump(pack, f, ensure_ascii=False, indent=2)
        
        with open(f'vocabs/{pack_name}_v1.msgpack', 'wb') as f:
            f.write(msgpack.packb(pack))
        
        print(f"✅ Created {pack_name} pack: {len(vocab)} tokens, {pack['size_mb']:.1f}MB")
    
    def _merge_corpora(self, languages, output_file):
        """Merge corpora from multiple languages"""
        with open(output_file, 'w', encoding='utf-8') as out:
            for lang in languages:
                corpus_file = f"{self.corpus_dir}/{lang}_corpus.txt"
                if os.path.exists(corpus_file):
                    with open(corpus_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            out.write(line)

# Create all vocabulary packs
creator = VocabularyPackCreator()
creator.create_all_packs()