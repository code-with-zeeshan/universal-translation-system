# data/download_training_data.py
import datasets
from datasets import load_dataset
from pathlib import Path
import logging
from typing import List
from concurrent.futures import ProcessPoolExecutor
import torch
from tqdm import tqdm

class MultilingualDataCollector:
    """Collect high-quality parallel data for 20 languages with modern practices"""
    
    def __init__(self, target_languages: List[str]):
        self.languages = target_languages
        log_dir = Path('log')
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_dir /'data_pipeline.log'), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
        
        self.data_sources = {
            'opus_datasets': [
                'Helsinki-NLP/opus-100',
                'Helsinki-NLP/opus_opensubtitles',
                'Helsinki-NLP/opus_multiun'
            ],
            'facebook_datasets': [
                'facebook/flores',
                'facebook/nllb-seed'
            ],
            'google_datasets': [
                'wmt19',
                'wmt20',
                'wmt21'
            ]
        }
    
    def download_all_data(self, output_dir: str = 'data/raw') -> None:
        """Download all available data with streaming and memory management"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self._download_huggingface_data(output_dir)
        self._download_opus_data(output_dir)
        self._download_wmt_data(output_dir)
        
        self.logger.info(f"✅ Downloaded all data to {output_dir}")
    
    def _download_huggingface_data(self, output_dir: Path) -> None:
        """Download from HuggingFace with streaming"""
        self.logger.info("📥 Downloading HuggingFace datasets...")
        
        # FLORES-200
        try:
            flores = load_dataset(
                'facebook/flores',
                name='flores200_sacrebleu_tokenized_xlm_roberta_base',
                split='dev',
                streaming=True,
                trust_remote_code=True
            )
            self._process_streaming_dataset(flores, output_dir / 'flores200')
        except Exception as e:
            self.logger.error(f"✗ Failed to download FLORES-200: {e}")
        
        # NLLB-MD
        for lang_pair in self._get_language_pairs():
            try:
                dataset = load_dataset(
                    'facebook/nllb-seed',
                    f"{lang_pair[0]}-{lang_pair[1]}",
                    streaming=True,
 n                    trust_remote_code=True
                )
                self._process_streaming_dataset(dataset, output_dir / f'nllb_{lang_pair[0]}_{lang_pair[1]}')
            except Exception as e:
                self.logger.error(f"✗ No NLLB data for {lang_pair}: {e}")
        
        # CCMatrix
        try:
            ccmatrix = load_dataset('yhavinga/ccmatrix', 'multilingual', streaming=True, trust_remote_code=True)
            self._process_streaming_dataset(ccmatrix, output_dir / 'ccmatrix')
        except Exception as e:
            self.logger.error(f"✗ Failed to download CCMatrix: {e}")
    
    def _download_opus_data(self, output_dir: Path) -> None:
        """Download OPUS datasets using HuggingFace"""
        opus_dir = output_dir / 'opus'
        opus_dir.mkdir(parents=True, exist_ok=True)
        
        for dataset_name in tqdm(self.data_sources['opus_datasets'], desc="Downloading OPUS"):
            try:
                dataset = load_dataset(dataset_name, streaming=True, trust_remote_code=True)
                self._process_streaming_dataset(dataset, opus_dir / dataset_name.split('/')[-1])
            except Exception as e:
                self.logger.error(f"✗ Failed to download {dataset_name}: {e}")
    
    def _download_wmt_data(self, output_dir: Path) -> None:
        """Download WMT datasets"""
        for dataset_name in self.data_sources['google_datasets']:
            try:
                dataset = load_dataset(dataset_name, streaming=True, trust_remote_code=True)
                self._process_streaming_dataset(dataset, output_dir / dataset_name)
            except Exception as e:
                self.logger.error(f"✗ Failed to download {dataset_name}: {e}")
    
    def _process_streaming_dataset(self, dataset, output_path: Path) -> None:
        """Process streaming dataset with memory efficiency"""
        output_path.mkdir(parents=True, exist_ok=True)
        for batch in dataset.iter(batch_size=1000):
            datasets.Dataset.from_dict(batch).save_to_disk(output_path)
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def _get_language_pairs(self) -> List[tuple]:
        """Generate language pairs"""
        pairs = []
        for lang1 in self.languages:
            for lang2 in self.languages:
                if lang1 < lang2:
                    pairs.append((lang1, lang2))
        return pairs

if __name__ == "__main__":
    collector = MultilingualDataCollector([
        'en', 'es', 'fr', 'de', 'zh', 'ja', 'ko', 'ar', 'hi', 'ru',
        'pt', 'it', 'tr', 'th', 'vi', 'pl', 'uk', 'nl', 'id', 'sv'
    ])
    collector.download_all_data()