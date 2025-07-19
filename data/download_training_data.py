# data/download_training_data.py (raw training data)
"""
Large-scale training data collector - Refactored to use shared utilities
Maintains backwards compatibility for standalone execution
"""

from pathlib import Path
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor
import torch
from tqdm import tqdm

# Import shared utilities with fallback
try:
    from data_utils import ConfigManager, DataProcessor, DatasetLoader, get_corpus_size_mb
    from utils.common_utils import StandardLogger, DirectoryManager
    INTEGRATED_MODE = True
except ImportError:
    # Fallback for standalone execution
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


class MultilingualDataCollector:
    """Collect high-quality parallel data for multiple languages with modern practices"""
    
    def __init__(self, target_languages: List[str] = None):
        # Initialize logger
        self.logger = StandardLogger.get_logger(__name__)
        
        # Get languages from config or use provided list
        if INTEGRATED_MODE and target_languages is None:
            self.languages = ConfigManager.get_languages()
            self.data_processor = DataProcessor(self.logger)
            self.dataset_loader = DatasetLoader(self.logger)
            self.training_distribution = ConfigManager.get_training_distribution()
        else:
            # Fallback for standalone mode
            self.languages = target_languages or [
                'en', 'es', 'fr', 'de', 'zh', 'ja', 'ko', 'ar', 'hi', 'ru',
                'pt', 'it', 'tr', 'th', 'vi', 'pl', 'uk', 'nl', 'id', 'sv'
            ]
            self.training_distribution = {}
        
        # Data source configuration
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
        
        self.logger.info(f"📊 Initialized collector for {len(self.languages)} languages")
    
    def download_all_data(self, output_dir: str = 'data/raw') -> Dict[str, int]:
        """
        Download all available data with streaming and memory management
        
        Returns:
            Dictionary with download statistics
        """
        output_path = DirectoryManager.create_directory(output_dir)
        
        stats = {
            'huggingface': 0,
            'opus': 0,
            'wmt': 0,
            'total_files': 0,
            'total_size_mb': 0
        }
        
        # Download from different sources
        stats['huggingface'] = self._download_huggingface_data(output_path)
        stats['opus'] = self._download_opus_data(output_path)
        stats['wmt'] = self._download_wmt_data(output_path)
        
        # Calculate totals
        stats['total_files'] = stats['huggingface'] + stats['opus'] + stats['wmt']
        
        # Calculate total size
        for file_path in output_path.rglob('*.txt'):
            if INTEGRATED_MODE:
                stats['total_size_mb'] += get_corpus_size_mb(file_path)
            else:
                stats['total_size_mb'] += file_path.stat().st_size / (1024 * 1024)
        
        self.logger.info(f"✅ Downloaded {stats['total_files']} datasets ({stats['total_size_mb']:.1f}MB)")
        return stats
    
    def _download_huggingface_data(self, output_dir: Path) -> int:
        """Download from HuggingFace with streaming"""
        self.logger.info("📥 Downloading HuggingFace datasets...")
        downloaded_count = 0
        
        # FLORES-200
        if INTEGRATED_MODE:
            dataset = self.dataset_loader.load_dataset_safely(
                'facebook/flores',
                config_name='flores200_sacrebleu_tokenized_xlm_roberta_base',
                split='dev',
                streaming=True
            )
            if dataset:
                self.data_processor.process_streaming_dataset(
                    dataset, output_dir / 'flores200'
                )
                downloaded_count += 1
        else:
            # Fallback implementation
            try:
                from datasets import load_dataset
                flores = load_dataset(
                    'facebook/flores',
                    name='flores200_sacrebleu_tokenized_xlm_roberta_base',
                    split='dev',
                    streaming=True,
                    trust_remote_code=True
                )
                self._process_streaming_dataset_fallback(flores, output_dir / 'flores200')
                downloaded_count += 1
            except Exception as e:
                self.logger.error(f"✗ Failed to download FLORES-200: {e}")
        
        # NLLB-MD for configured language pairs
        for lang_pair in self._get_language_pairs():
            pair_str = f"{lang_pair[0]}-{lang_pair[1]}"
            
            # Skip if not in training distribution (when integrated)
            if INTEGRATED_MODE and self.training_distribution:
                if pair_str not in self.training_distribution:
                    continue
            
            try:
                if INTEGRATED_MODE:
                    dataset = self.dataset_loader.load_dataset_safely(
                        'facebook/nllb-seed',
                        config_name=pair_str,
                        streaming=True
                    )
                    if dataset:
                        target_size = self.training_distribution.get(pair_str, 100000)
                        self.data_processor.process_streaming_dataset(
                            dataset,
                            output_dir / f'nllb_{pair_str}',
                            max_samples=target_size
                        )
                        downloaded_count += 1
                else:
                    # Fallback implementation
                    from datasets import load_dataset
                    dataset = load_dataset(
                        'facebook/nllb-seed',
                        pair_str,
                        streaming=True,
                        trust_remote_code=True
                    )
                    self._process_streaming_dataset_fallback(
                        dataset, output_dir / f'nllb_{pair_str}'
                    )
                    downloaded_count += 1
            except Exception as e:
                self.logger.debug(f"No NLLB data for {pair_str}: {e}")
        
        # CCMatrix
        try:
            if INTEGRATED_MODE:
                dataset = self.dataset_loader.load_dataset_safely(
                    'yhavinga/ccmatrix',
                    config_name='multilingual',
                    streaming=True
                )
                if dataset:
                    self.data_processor.process_streaming_dataset(
                        dataset, output_dir / 'ccmatrix'
                    )
                    downloaded_count += 1
            else:
                from datasets import load_dataset
                ccmatrix = load_dataset(
                    'yhavinga/ccmatrix', 'multilingual', streaming=True, trust_remote_code=True
                )
                self._process_streaming_dataset_fallback(ccmatrix, output_dir / 'ccmatrix')
                downloaded_count += 1
        except Exception as e:
            self.logger.error(f"✗ Failed to download CCMatrix: {e}")
        
        return downloaded_count
    
    def _download_opus_data(self, output_dir: Path) -> int:
        """Download OPUS datasets using HuggingFace"""
        opus_dir = DirectoryManager.create_directory(output_dir / 'opus')
        downloaded_count = 0
        
        for dataset_name in tqdm(self.data_sources['opus_datasets'], desc="Downloading OPUS"):
            try:
                if INTEGRATED_MODE:
                    dataset = self.dataset_loader.load_dataset_safely(
                        dataset_name,
                        streaming=True
                    )
                    if dataset:
                        self.data_processor.process_streaming_dataset(
                            dataset,
                            opus_dir / dataset_name.split('/')[-1]
                        )
                        downloaded_count += 1
                else:
                    from datasets import load_dataset
                    dataset = load_dataset(dataset_name, streaming=True, trust_remote_code=True)
                    self._process_streaming_dataset_fallback(
                        dataset, opus_dir / dataset_name.split('/')[-1]
                    )
                    downloaded_count += 1
            except Exception as e:
                self.logger.error(f"✗ Failed to download {dataset_name}: {e}")
        
        return downloaded_count
    
    def _download_wmt_data(self, output_dir: Path) -> int:
        """Download WMT datasets"""
        wmt_dir = DirectoryManager.create_directory(output_dir / 'wmt')
        downloaded_count = 0
        
        for dataset_name in self.data_sources['google_datasets']:
            try:
                if INTEGRATED_MODE:
                    dataset = self.dataset_loader.load_dataset_safely(
                        dataset_name,
                        streaming=True
                    )
                    if dataset:
                        self.data_processor.process_streaming_dataset(
                            dataset,
                            wmt_dir / dataset_name
                        )
                        downloaded_count += 1
                else:
                    from datasets import load_dataset
                    dataset = load_dataset(dataset_name, streaming=True, trust_remote_code=True)
                    self._process_streaming_dataset_fallback(
                        dataset, wmt_dir / dataset_name
                    )
                    downloaded_count += 1
            except Exception as e:
                self.logger.error(f"✗ Failed to download {dataset_name}: {e}")
        
        return downloaded_count
    
    def _process_streaming_dataset_fallback(self, dataset, output_path: Path) -> None:
        """Fallback processing for standalone mode"""
        from datasets import Dataset
        
        DirectoryManager.create_directory(output_path)
        
        batch_data = []
        for i, sample in enumerate(dataset):
            batch_data.append(sample)
            
            if len(batch_data) >= 1000:
                Dataset.from_list(batch_data).save_to_disk(
                    str(output_path / f"batch_{i}")
                )
                batch_data = []
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Save remaining data
        if batch_data:
            Dataset.from_list(batch_data).save_to_disk(
                str(output_path / "batch_final")
            )
    
    def _get_language_pairs(self) -> List[tuple]:
        """Generate language pairs based on configuration or defaults"""
        pairs = []
        
        if INTEGRATED_MODE and self.training_distribution:
            # Use configured pairs
            for pair_str in self.training_distribution.keys():
                if '-' in pair_str:
                    source, target = pair_str.split('-')
                    pairs.append((source, target))
        else:
            # Generate all possible pairs for standalone mode
            for i, lang1 in enumerate(self.languages):
                for lang2 in self.languages[i+1:]:
                    pairs.append((lang1, lang2))
        
        return pairs
    
    def download_specific_pair(self, source: str, target: str, output_dir: str = 'data/raw') -> bool:
        """
        Download data for a specific language pair
        
        Args:
            source: Source language code
            target: Target language code
            output_dir: Output directory
            
        Returns:
            True if successful
        """
        output_path = DirectoryManager.create_directory(output_dir)
        pair_str = f"{source}-{target}"
        
        self.logger.info(f"📥 Downloading data for {pair_str}")
        
        success = False
        
        # Try different sources
        for dataset_name in ['Helsinki-NLP/opus-100', 'facebook/nllb-seed']:
            try:
                if INTEGRATED_MODE:
                    dataset = self.dataset_loader.load_dataset_safely(
                        dataset_name,
                        config_name=pair_str,
                        streaming=True
                    )
                    if dataset:
                        self.data_processor.process_streaming_dataset(
                            dataset,
                            output_path / f"{dataset_name.split('/')[-1]}_{pair_str}"
                        )
                        success = True
                        break
                else:
                    from datasets import load_dataset
                    dataset = load_dataset(
                        dataset_name,
                        pair_str,
                        streaming=True,
                        trust_remote_code=True
                    )
                    self._process_streaming_dataset_fallback(
                        dataset,
                        output_path / f"{dataset_name.split('/')[-1]}_{pair_str}"
                    )
                    success = True
                    break
            except Exception as e:
                self.logger.debug(f"No data in {dataset_name} for {pair_str}: {e}")
        
        if success:
            self.logger.info(f"✅ Successfully downloaded data for {pair_str}")
        else:
            self.logger.warning(f"⚠️  No data found for {pair_str}")
        
        return success


def main():
    """Main entry point for standalone execution"""
    # When run standalone, use default languages
    collector = MultilingualDataCollector([
        'en', 'es', 'fr', 'de', 'zh', 'ja', 'ko', 'ar', 'hi', 'ru',
        'pt', 'it', 'tr', 'th', 'vi', 'pl', 'uk', 'nl', 'id', 'sv'
    ])
    
    # Download all data
    stats = collector.download_all_data()
    
    print(f"\nDownload complete! Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()