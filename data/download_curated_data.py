# data/download_curated_data.py
import datasets
import os
from typing import Dict, List

class CuratedDataDownloader:
    """Download only high-quality, manageable datasets"""
    
    def __init__(self):
        self.data_sources = {
            # Small, high-quality datasets
            'flores200': {
                'dataset': 'facebook/flores',
                'size': '~10MB per language',
                'sentences': 1000,  # Dev + test
                'quality': 'excellent'
            },
            
            'tatoeba': {
                'dataset': 'Helsinki-NLP/tatoeba',
                'size': '~5-50MB per pair',
                'sentences': '1k-500k',
                'quality': 'good'
            },
            
            'opus_books': {
                'dataset': 'opus_books',
                'size': '~100MB per pair',
                'sentences': '10k-100k',
                'quality': 'excellent'
            },
            
            'ted_talks': {
                'dataset': 'ted_talks_iwslt',
                'size': '~50MB',
                'sentences': '200k',
                'quality': 'excellent'
            }
        }
    
    def download_essential_data(self, output_dir='data/essential'):
        """Download only essential, high-quality data"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. FLORES-200 for evaluation (MUST HAVE)
        print("📥 Downloading FLORES-200 (10MB)...")
        flores = datasets.load_dataset('facebook/flores', 'all')
        flores.save_to_disk(f"{output_dir}/flores200")
        
        # 2. Tatoeba for diverse examples
        print("📥 Downloading Tatoeba sentences...")
        for lang_pair in self.get_english_pairs():
            try:
                dataset = datasets.load_dataset(
                    'Helsinki-NLP/tatoeba',
                    lang1='eng',
                    lang2=lang_pair,
                    split='train'
                )
                
                # Take only first 100k sentences to keep size manageable
                dataset = dataset.select(range(min(100000, len(dataset))))
                dataset.save_to_disk(f"{output_dir}/tatoeba_en_{lang_pair}")
                
                print(f"  ✓ en-{lang_pair}: {len(dataset)} sentences")
            except:
                print(f"  ✗ en-{lang_pair}: Not available")
        
        # 3. OpenSubtitles sample (informal language)
        print("📥 Downloading OpenSubtitles sample...")
        self.download_opus_sample('OpenSubtitles', output_dir, max_size_mb=100)
        
        # 4. MultiUN sample (formal language)  
        print("📥 Downloading MultiUN sample...")
        self.download_opus_sample('MultiUN', output_dir, max_size_mb=100)
    
    def download_opus_sample(self, corpus_name, output_dir, max_size_mb=100):
        """Download sample from OPUS (not entire corpus)"""
        import requests
        import gzip
        
        base_url = f"https://object.pouta.csc.fi/OPUS-{corpus_name}/v2018/moses"
        
        for lang_pair in ['en-es', 'en-fr', 'en-de', 'en-zh']:
            try:
                url = f"{base_url}/{lang_pair}.txt.zip"
                
                # Download with size limit
                response = requests.get(url, stream=True)
                
                size = 0
                chunk_size = 1024 * 1024  # 1MB chunks
                
                output_file = f"{output_dir}/{corpus_name}_{lang_pair}.txt"
                
                with open(output_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        size += len(chunk)
                        
                        if size > max_size_mb * 1024 * 1024:
                            break
                
                print(f"  ✓ {lang_pair}: {size / 1024 / 1024:.1f}MB")
                
            except Exception as e:
                print(f"  ✗ {lang_pair}: {str(e)}")

# Download essential data only
downloader = CuratedDataDownloader()
downloader.download_essential_data()