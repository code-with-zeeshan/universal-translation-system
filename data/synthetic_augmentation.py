# data/synthetic_augmentation.py
class SyntheticDataAugmenter:
    """Generate additional training data using existing models"""
    
    def __init__(self, base_model='facebook/nllb-200-distilled-600M'):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
    def augment_with_backtranslation(
        self,
        monolingual_file: str,
        source_lang: str,
        target_lang: str,
        output_file: str
    ):
        """Use backtranslation to create synthetic parallel data"""
        
        with open(monolingual_file, 'r') as f_in:
            with open(output_file, 'w') as f_out:
                for line in f_in:
                    text = line.strip()
                    
                    # Translate to target language
                    translated = self.translate(text, source_lang, target_lang)
                    
                    # Translate back to source
                    back_translated = self.translate(translated, target_lang, source_lang)
                    
                    # If back-translation is similar, it's good quality
                    if self.is_similar(text, back_translated):
                        f_out.write(f"{text}\t{translated}\n")
    
    def generate_pivot_translations(self, english_pairs_dir: str):
        """Generate non-English pairs using English as pivot"""
        
        # If we have en->es and en->fr, we can create es->fr
        # This is lower quality but helps with zero-shot pairs
        
        pairs_data = {}
        
        # Load all English-centric pairs
        for lang in ['es', 'fr', 'de', 'zh', 'ja']:
            pairs_data[lang] = self.load_pairs(f"{english_pairs_dir}/en-{lang}.txt")
        
        # Generate pivoted pairs
        for lang1 in pairs_data:
            for lang2 in pairs_data:
                if lang1 < lang2:  # Avoid duplicates
                    self.create_pivot_pairs(
                        pairs_data[lang1],
                        pairs_data[lang2],
                        lang1,
                        lang2
                    )