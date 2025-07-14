# data/download_curated_data.py
import datasets
from datasets import load_dataset
from pathlib import Path
import logging
from typing import Dict, List
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

class CuratedDataDownloader:
    """Download high-quality, manageable datasets (~100MB total) with modern practices"""
    
    def __init__(self):
        # Configure logging
        log_dir = Path('log')
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_dir /'data_pipeline.log'), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
        
        # Modern configuration for datasets
        self.data_sources: Dict[str, Dict] = {
            'flores200': {
                'dataset_name': 'facebook/flores',
                'config_name': 'flores200_sacrebleu_tokenized_xlm_roberta_base',
                'split': 'dev',
                'size': '~10MB per language',
                'sentences': 1000,
                'quality': 'excellent'
            },
            'tatoeba': {
                'dataset_name': 'Helsinki-NLP/tatoeba',
                'size': '~5-50MB per pair',
                'sentences': '1k-500k',
                'quality': 'good'
            },
            'opus_books': {
                'dataset_name': 'opus_books',
                'size': '~100MB per pair',
                'sentences': '10k-100k',
                'quality': 'excellent'
            },
            'ted_talks': {
                'dataset_name': 'ted_talks_iwslt',
                'size': '~50MB',
                'sentences': '200k',
                'quality': 'excellent'
            }
        }
        
        # Configure requests session with retries
        self.session = requests.Session()
        retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def get_english_pairs(self) -> List[str]:
        """Return list of target languages for English-centric pairs"""
        return ['es', 'fr', 'de', 'zh', 'ja', 'ko', 'ar', 'hi', 'ru', 'pt', 'it', 'tr', 'th', 'vi', 'pl', 'uk', 'nl', 'id', 'sv']
    
    def download_essential_data(self, output_dir: str = 'data/essential') -> None:
        """Download essential, high-quality data with modern HuggingFace API"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. FLORES-200 for evaluation
        self.logger.info("📥 Downloading FLORES-200...")
        try:
            flores = load_dataset(
                self.data_sources['flores200']['dataset_name'],
                name=self.data_sources['flores200']['config_name'],
                split=self.data_sources['flores200']['split'],
                trust_remote_code=True
            )
            flores.save_to_disk(output_dir / 'flores200')
            self.logger.info(f"✓ FLORES-200 saved to {output_dir / 'flores200'}")
        except Exception as e:
            self.logger.error(f"✗ Failed to download FLORES-200: {e}")
        
        # 2. Tatoeba for diverse examples
        self.logger.info("📥 Downloading Tatoeba sentences...")
        for lang_pair in tqdm(self.get_english_pairs(), desc="Downloading Tatoeba"):
            try:
                dataset = load_dataset(
                    self.data_sources['tatoeba']['dataset_name'],
                    lang1='eng',
                    lang2=lang_pair,
                    split='train',
                    trust_remote_code=True
                )
                dataset = dataset.select(range(min(100000, len(dataset))))
                dataset.save_to_disk(output_dir / f'tatoeba_en_{lang_pair}')
                self.logger.info(f"✓ en-{lang_pair}: {len(dataset)} sentences")
            except Exception as e:
                self.logger.error(f"✗ en-{lang_pair}: {e}")
        
        # 3. OpenSubtitles sample
        self.logger.info("📥 Downloading OpenSubtitles sample...")
        self.download_opus_sample('OpenSubtitles', output_dir, max_size_mb=100)
        
        # 4. MultiUN sample
        self.logger.info("📥 Downloading MultiUN sample...")
        self.download_opus_sample('MultiUN', output_dir, max_size_mb=100)
    
    def download_opus_sample(self, corpus_name: str, output_dir: Path, max_size_mb: int = 100) -> None:
        """Download sample from OPUS using modern requests with streaming"""
        base_url = f"https://object.pouta.csc.fi/OPUS-{corpus_name}/v2018/moses"
        
        for lang_pair in ['en-es', 'en-fr', 'en-de', 'en-zh']:
            try:
                url = f"{base_url}/{lang_pair}.txt.zip"
                output_file = output_dir / f'{corpus_name}_{lang_pair}.txt'
                
                size = 0
                chunk_size = 1024 * 1024  # 1MB chunks
                
                with self.session.get(url, stream=True, timeout=30) as response:
                    response.raise_for_status()
                    with open(output_file, 'wb') as f:
                        for chunk in tqdm(response.iter_content(chunk_size=chunk_size), desc=f"Downloading {lang_pair}"):
                            f.write(chunk)
                            size += len(chunk)
                            if size > max_size_mb * 1024 * 1024:
                                break
                self.logger.info(f"✓ {lang_pair}: {size / 1024 / 1024:.1f}MB")
            except Exception as e:
                self.logger.error(f"✗ {lang_pair}: {e}")