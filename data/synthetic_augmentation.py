# data/synthetic_augmentation.py
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch
from pathlib import Path
import logging
from typing import Dict, List
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer, util

class SyntheticDataAugmenter:
    """Generate additional training data using modern transformer models"""
    
    def __init__(self, base_model: str = 'facebook/nllb-200-distilled-1.3B'):
        log_dir = Path('log')
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_dir /'data_pipeline.log'), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.translator = pipeline(
            "translation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        # Initialize sentence transformer for similarity checking
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def augment_with_backtranslation(
        self, monolingual_file: str, source_lang: str, target_lang: str, output_file: str
    ) -> None:
        """Use backtranslation to create synthetic parallel data"""
        monolingual_file = Path(monolingual_file)
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"📝 Generating backtranslations for {source_lang}->{target_lang}")
        with open(monolingual_file, 'r', encoding='utf-8') as f_in:
            with open(output_file, 'w', encoding='utf-8') as f_out:
                for line in tqdm(f_in, desc="Backtranslating"):
                    text = line.strip()
                    if not text:
                        continue
                    
                    # Translate to target language
                    try:
                        translated = self.translator(
                            text,
                            src_lang=source_lang,
                            tgt_lang=target_lang,
                            max_length=512
                        )[0]['translation_text']
                        
                        # Translate back to source
                        back_translated = self.translator(
                            translated,
                            src_lang=target_lang,
                            tgt_lang=source_lang,
                            max_length=512
                        )[0]['translation_text']
                        
                        # Check similarity
                        if self.is_similar(text, back_translated):
                            f_out.write(f"{text}\t{translated}\n")
                    except Exception as e:
                        self.logger.error(f"✗ Failed to process line: {e}")
    
    def generate_pivot_translations(self, english_pairs_dir: str) -> None:
        """Generate non-English pairs using English as pivot"""
        english_pairs_dir = Path(english_pairs_dir)
        pairs_data: Dict[str, List] = {}
        
        self.logger.info("📥 Loading English-centric pairs...")
        for lang in ['es', 'fr', 'de', 'zh', 'ja']:
            try:
                pairs_data[lang] = self.load_pairs(english_pairs_dir / f'en-{lang}.txt')
            except Exception as e:
                self.logger.error(f"✗ Failed to load en-{lang}: {e}")
        
        self.logger.info("📝 Generating pivoted pairs...")
        for lang1 in pairs_data:
            for lang2 in pairs_data:
                if lang1 < lang2:
                    try:
                        self.create_pivot_pairs(pairs_data[lang1], pairs_data[lang2], lang1, lang2)
                    except Exception as e:
                        self.logger.error(f"✗ Failed to create {lang1}-{lang2}: {e}")
    
    def load_pairs(self, file_path: Path) -> List[tuple]:
        """Load sentence pairs from file"""
        pairs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    pairs.append(parts)
        return pairs
    
    def create_pivot_pairs(self, pairs1: List[tuple], pairs2: List[tuple], lang1: str, lang2: str) -> None:
        """Create pivot translations"""
        self.logger.info(f"Creating pivot pairs for {lang1}-{lang2}")
        output_file = Path(f'data/final/pivot_pairs/{lang1}-{lang2}.txt')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for (en_text1, text1), (en_text2, text2) in zip(pairs1, pairs2):
                if en_text1 == en_text2:  # Match sentences via English pivot
                    try:
                        f_out.write(f"{text1}\t{text2}\n")
                    except Exception as e:
                        self.logger.error(f"✗ Failed to write {lang1}-{lang2} pair: {e}")
    
    def is_similar(self, text1: str, text2: str) -> bool:
        """Check if back-translated text is similar to original using sentence embeddings"""
        try:
            embeddings = self.sentence_model.encode([text1, text2], convert_to_tensor=True)
            similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
            threshold = 0.85  # Adjustable similarity threshold
            self.logger.debug(f"Similarity score: {similarity:.3f} for texts: {text1[:50]}... vs {text2[:50]}...")
            return similarity >= threshold
        except Exception as e:
            self.logger.error(f"✗ Failed to compute similarity: {e}")
            return False