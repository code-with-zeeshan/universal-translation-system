# data/download_training_data.py
import datasets
import requests
import os
from typing import Dict, List

class MultilingualDataCollector:
    """Collect high-quality parallel data for 20 languages"""
    
    def __init__(self, target_languages: List[str]):
        self.languages = target_languages
        self.data_sources = {
            # High-quality parallel corpora
            'opus': {
                'url': 'https://opus.nlpl.eu/download.php',
                'datasets': [
                    'OpenSubtitles',  # Movie subtitles (informal)
                    'MultiUN',        # UN documents (formal)
                    'WikiMatrix',     # Wikipedia parallel sentences
                    'CCAligned',      # Common Crawl aligned
                    'ParaCrawl',      # Web crawl parallel data
                    'TED2020',        # TED talks
                    'Tatoeba',        # Community translations
                ]
            },
            
            # Facebook's datasets
            'facebook': {
                'flores200': 'facebook/flores',  # High quality eval set
                'nllb-seed': 'allenai/nllb',     # NLLB training data
                'ccmatrix': 'yhavinga/ccmatrix'  # Mining from CommonCrawl
            },
            
            # Google's datasets  
            'google': {
                'wmt': 'wmt19','wmt20','wmt21',  # Translation competitions
                'c4-multilingual': 'mc4'            # Multilingual C4
            }
        }
    
    def download_all_data(self, output_dir='data/raw'):
        """Download all available data"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Download from HuggingFace datasets
        self._download_huggingface_data(output_dir)
        
        # 2. Download OPUS data
        self._download_opus_data(output_dir)
        
        # 3. Download WMT datasets
        self._download_wmt_data(output_dir)
        
        print(f"✅ Downloaded all data to {output_dir}")
        
    def _download_huggingface_data(self, output_dir):
        """Download from HuggingFace"""
        
        # FLORES-200 (high quality dev/test)
        flores = datasets.load_dataset("facebook/flores", "all")
        flores.save_to_disk(f"{output_dir}/flores200")
        
        # NLLB-MD (filtered from NLLB)
        for lang_pair in self._get_language_pairs():
            try:
                dataset = datasets.load_dataset(
                    "allenai/nllb", 
                    f"{lang_pair[0]}-{lang_pair[1]}"
                )
                dataset.save_to_disk(f"{output_dir}/nllb_{lang_pair[0]}_{lang_pair[1]}")
            except:
                print(f"No NLLB data for {lang_pair}")
        
        # CCMatrix
        ccmatrix = datasets.load_dataset("yhavinga/ccmatrix", "multilingual")
        ccmatrix.save_to_disk(f"{output_dir}/ccmatrix")
    
    def _download_opus_data(self, output_dir):
        """Download OPUS datasets"""
        import opustools
        
        opus_dir = f"{output_dir}/opus"
        os.makedirs(opus_dir, exist_ok=True)
        
        for lang1 in self.languages:
            for lang2 in self.languages:
                if lang1 < lang2:  # Avoid duplicates
                    try:
                        # Download parallel corpus
                        opus_reader = opustools.OpusRead(
                            directory="OpenSubtitles",
                            source=lang1,
                            target=lang2,
                            write=f"{opus_dir}/{lang1}-{lang2}.txt",
                            download_dir=opus_dir
                        )
                        opus_reader.printPairs()
                    except:
                        print(f"No OPUS data for {lang1}-{lang2}")

# Run data collection
collector = MultilingualDataCollector([
    'en', 'es', 'fr', 'de', 'zh', 'ja', 'ko', 'ar', 'hi', 'ru',
    'pt', 'it', 'tr', 'th', 'vi', 'pl', 'uk', 'nl', 'id', 'sv'
])
collector.download_all_data()