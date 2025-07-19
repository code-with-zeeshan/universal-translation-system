# data/synthetic_augmentation.py
"""
Synthetic data augmentation - Refactored to use shared utilities
Generate additional training data using modern transformer models
"""

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Import shared utilities with fallback
try:
    from data_utils import ConfigManager, estimate_sentence_count
    from utils.common_utils import StandardLogger, DirectoryManager
    INTEGRATED_MODE = True
except ImportError:
    import logging
    INTEGRATED_MODE = False
    
    class StandardLogger:
        @staticmethod
        def get_logger(name):
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            return logging.getLogger(name)
    
    class DirectoryManager:
        @staticmethod
        def create_directory(path):
            Path(path).mkdir(parents=True, exist_ok=True)
            return Path(path)


class SyntheticDataAugmenter:
    """Generate additional training data using modern transformer models"""
    
    def __init__(self, base_model: str = 'facebook/nllb-200-distilled-1.3B'):
        self.logger = StandardLogger.get_logger(__name__)
        self.base_model = base_model
        
        # Get configuration if available
        if INTEGRATED_MODE:
            self.languages = ConfigManager.get_languages()
            self.quality_threshold = ConfigManager.get_quality_threshold()
            self.output_dir = Path(ConfigManager.get_output_dir())
        else:
            self.languages = ['en', 'es', 'fr', 'de', 'zh', 'ja', 'ko', 'ar', 'hi', 'ru',
                             'pt', 'it', 'tr', 'th', 'vi', 'pl', 'uk', 'nl', 'id', 'sv']
            self.quality_threshold = 0.8
            self.output_dir = Path('data/processed')
        
        # Initialize models lazily
        self._model = None
        self._tokenizer = None
        self._translator = None
        self._sentence_model = None
        
        self.logger.info(f"📊 Initialized augmenter with model: {base_model}")
    
    @property
    def model(self):
        """Lazy load translation model"""
        if self._model is None:
            self.logger.info("🔄 Loading translation model...")
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
        return self._model
    
    @property
    def tokenizer(self):
        """Lazy load tokenizer"""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        return self._tokenizer
    
    @property
    def translator(self):
        """Lazy load translator pipeline"""
        if self._translator is None:
            self._translator = pipeline(
                "translation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                batch_size=8 if torch.cuda.is_available() else 1
            )
        return self._translator
    
    @property
    def sentence_model(self):
        """Lazy load sentence transformer"""
        if self._sentence_model is None:
            self.logger.info("🔄 Loading sentence transformer...")
            self._sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._sentence_model
    
    def augment_with_backtranslation(
        self,
        monolingual_file: str,
        source_lang: str,
        target_lang: str,
        output_file: str,
        max_sentences: int = 100000,
        batch_size: int = 32
    ) -> Dict[str, int]:
        """
        Use backtranslation to create synthetic parallel data
        
        Returns:
            Statistics about the augmentation process
        """
        monolingual_path = Path(monolingual_file)
        output_path = Path(output_file)
        DirectoryManager.create_directory(output_path.parent)
        
        if not monolingual_path.exists():
            self.logger.error(f"❌ Monolingual file not found: {monolingual_path}")
            return {'error': 'file_not_found', 'augmented': 0}
        
        self.logger.info(f"📝 Generating backtranslations for {source_lang}->{target_lang}")
        
        # Estimate total sentences
        if INTEGRATED_MODE:
            total_sentences = estimate_sentence_count(monolingual_path)
        else:
            total_sentences = sum(1 for _ in open(monolingual_path, 'r', encoding='utf-8'))
        
        sentences_to_process = min(total_sentences, max_sentences)
        
        stats = {
            'total_sentences': total_sentences,
            'processed': 0,
            'augmented': 0,
            'filtered_quality': 0,
            'errors': 0
        }
        
        # Process in batches
        with open(monolingual_path, 'r', encoding='utf-8') as f_in:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                batch_texts = []
                
                for line_num, line in enumerate(tqdm(f_in, total=sentences_to_process, 
                                                   desc="Backtranslating")):
                    if line_num >= sentences_to_process:
                        break
                    
                    text = line.strip()
                    if not text or len(text) < 10:
                        continue
                    
                    batch_texts.append(text)
                    
                    # Process batch when full
                    if len(batch_texts) >= batch_size:
                        results = self._process_backtranslation_batch(
                            batch_texts, source_lang, target_lang
                        )
                        
                        for original, translated, back_translated in results:
                            if translated and self._is_quality_translation(
                                original, back_translated
                            ):
                                f_out.write(f"{original}\t{translated}\n")
                                stats['augmented'] += 1
                            else:
                                stats['filtered_quality'] += 1
                        
                        stats['processed'] += len(batch_texts)
                        batch_texts = []
                
                # Process remaining batch
                if batch_texts:
                    results = self._process_backtranslation_batch(
                        batch_texts, source_lang, target_lang
                    )
                    
                    for original, translated, back_translated in results:
                        if translated and self._is_quality_translation(
                            original, back_translated
                        ):
                            f_out.write(f"{original}\t{translated}\n")
                            stats['augmented'] += 1
                        else:
                            stats['filtered_quality'] += 1
                    
                    stats['processed'] += len(batch_texts)
        
        self.logger.info(f"✅ Augmentation complete: {stats['augmented']:,} pairs created")
        self.logger.info(f"📊 Quality filtered: {stats['filtered_quality']:,} pairs")
        
        return stats
    
    def _process_backtranslation_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str
    ) -> List[Tuple[str, str, str]]:
        """Process a batch of texts for backtranslation"""
        results = []
        
        try:
            # Translate to target language
            translations = self.translator(
                texts,
                src_lang=self._get_nllb_code(source_lang),
                tgt_lang=self._get_nllb_code(target_lang),
                max_length=512
            )
            
            translated_texts = [t['translation_text'] for t in translations]
            
            # Translate back to source
            back_translations = self.translator(
                translated_texts,
                src_lang=self._get_nllb_code(target_lang),
                tgt_lang=self._get_nllb_code(source_lang),
                max_length=512
            )
            
            back_translated_texts = [t['translation_text'] for t in back_translations]
            
            # Combine results
            for original, translated, back_translated in zip(
                texts, translated_texts, back_translated_texts
            ):
                results.append((original, translated, back_translated))
                
        except Exception as e:
            self.logger.error(f"❌ Batch translation failed: {e}")
            # Return empty translations for failed batch
            results = [(text, None, None) for text in texts]
        
        return results
    
    def _get_nllb_code(self, lang_code: str) -> str:
        """Convert language code to NLLB format"""
        # NLLB uses specific language codes
        nllb_mapping = {
            'en': 'eng_Latn',
            'es': 'spa_Latn',
            'fr': 'fra_Latn',
            'de': 'deu_Latn',
            'zh': 'zho_Hans',
            'ja': 'jpn_Jpan',
            'ko': 'kor_Hang',
            'ar': 'arb_Arab',
            'hi': 'hin_Deva',
            'ru': 'rus_Cyrl',
            'pt': 'por_Latn',
            'it': 'ita_Latn',
            'tr': 'tur_Latn',
            'th': 'tha_Thai',
            'vi': 'vie_Latn',
            'pl': 'pol_Latn',
            'uk': 'ukr_Cyrl',
            'nl': 'nld_Latn',
            'id': 'ind_Latn',
            'sv': 'swe_Latn'
        }
        return nllb_mapping.get(lang_code, lang_code)
    
    def _is_quality_translation(self, original: str, back_translated: str) -> bool:
        """Check translation quality using similarity"""
        if not back_translated:
            return False
        
        try:
            # Compute embeddings
            embeddings = self.sentence_model.encode(
                [original, back_translated],
                convert_to_tensor=True
            )
            
            # Calculate cosine similarity
            similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
            
            return similarity >= self.quality_threshold
            
        except Exception as e:
            self.logger.error(f"❌ Quality check failed: {e}")
            return False
    
    def generate_pivot_translations(
        self,
        english_pairs_dir: str,
        output_dir: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Generate non-English pairs using English as pivot
        
        Returns:
            Statistics about pivot generation
        """
        english_pairs_path = Path(english_pairs_dir)
        if output_dir is None:
            output_path = self.output_dir / 'pivot_pairs'
        else:
            output_path = Path(output_dir)
        
        DirectoryManager.create_directory(output_path)
        
        # Load English-centric pairs
        pairs_data: Dict[str, List[Tuple[str, str]]] = {}
        
        self.logger.info("📥 Loading English-centric pairs...")
        
        # Load available language pairs
        for lang in self.languages:
            if lang == 'en':
                continue
                
            # Try different file patterns
            patterns = [
                f'en-{lang}_sampled.txt',
                f'en-{lang}.txt',
                f'opus_en-{lang}.txt'
            ]
            
            for pattern in patterns:
                file_path = english_pairs_path / pattern
                if file_path.exists():
                    pairs_data[lang] = self._load_pairs(file_path)
                    self.logger.info(f"✅ Loaded en-{lang}: {len(pairs_data[lang]):,} pairs")
                    break
        
        # Generate pivot pairs
        stats = {
            'total_pivot_pairs': 0,
            'pairs_created': {}
        }
        
        self.logger.info("📝 Generating pivoted pairs...")
        
        for lang1 in pairs_data:
            for lang2 in pairs_data:
                if lang1 < lang2:  # Avoid duplicates
                    pair_count = self._create_pivot_pairs(
                        pairs_data[lang1],
                        pairs_data[lang2],
                        lang1,
                        lang2,
                        output_path
                    )
                    
                    stats['pairs_created'][f'{lang1}-{lang2}'] = pair_count
                    stats['total_pivot_pairs'] += pair_count
        
        self.logger.info(f"✅ Generated {stats['total_pivot_pairs']:,} pivot pairs")
        
        return stats
    
    def _load_pairs(self, file_path: Path, max_pairs: int = 50000) -> List[Tuple[str, str]]:
        """Load sentence pairs from file"""
        pairs = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_pairs:
                    break
                    
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    pairs.append((parts[0], parts[1]))
        
        return pairs
    
    def _create_pivot_pairs(
        self,
        pairs1: List[Tuple[str, str]],
        pairs2: List[Tuple[str, str]],
        lang1: str,
        lang2: str,
        output_path: Path
    ) -> int:
        """Create pivot translations using English as intermediary"""
        self.logger.info(f"🔄 Creating pivot pairs for {lang1}-{lang2}")
        
        output_file = output_path / f'{lang1}-{lang2}_pivot.txt'
        
        # Create mapping from English to target languages
        en_to_lang1 = {en: lang for en, lang in pairs1}
        en_to_lang2 = {en: lang for en, lang in pairs2}
        
        # Find common English sentences
        common_en = set(en_to_lang1.keys()) & set(en_to_lang2.keys())
        
        pairs_created = 0
        
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for en_text in common_en:
                try:
                    lang1_text = en_to_lang1[en_text]
                    lang2_text = en_to_lang2[en_text]
                    
                    f_out.write(f"{lang1_text}\t{lang2_text}\n")
                    pairs_created += 1
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to create pivot pair: {e}")
        
        self.logger.info(f"✅ Created {pairs_created:,} {lang1}-{lang2} pairs")
        
        return pairs_created


def main():
    """Main entry point for standalone execution"""
    augmenter = SyntheticDataAugmenter()
    
    # Example: Augment English-Spanish with backtranslation
    print("🚀 Running synthetic data augmentation example...")
    
    # Create sample monolingual data
    sample_dir = Path("test_data")
    DirectoryManager.create_directory(sample_dir)
    
    sample_mono = sample_dir / "mono_en.txt"
    with open(sample_mono, 'w', encoding='utf-8') as f:
        f.write("This is a test sentence.\n")
        f.write("Machine translation is improving rapidly.\n")
        f.write("Neural networks have revolutionized NLP.\n")
    
    # Run augmentation
    stats = augmenter.augment_with_backtranslation(
        monolingual_file=str(sample_mono),
        source_lang='en',
        target_lang='es',
        output_file=str(sample_dir / "augmented_en-es.txt"),
        max_sentences=3
    )
    
    print(f"\nAugmentation statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()