# data/download_curated_data.py (evaluation data)
"""
Curated data downloader - Updated to use shared utilities
Maintains backwards compatibility for standalone execution
"""

import datasets
from datasets import load_dataset
from pathlib import Path
from typing import Dict, List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
import zipfile
import io

# Import shared utilities
try:
    from data_utils import ConfigManager, DataProcessor, DatasetLoader
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


class CuratedDataDownloader:
    """Download high-quality, manageable datasets with modern practices"""
    
    def __init__(self):
        # Use shared logger if available
        self.logger = StandardLogger.get_logger(__name__)
        
        # Use shared config if available
        if INTEGRATED_MODE:
            self.languages = ConfigManager.get_languages()
            self.dataset_loader = DatasetLoader(self.logger)
            self.data_processor = DataProcessor(self.logger)
        else:
            # Fallback configuration for standalone mode
            self.languages = ['en', 'es', 'fr', 'de', 'zh', 'ja', 'ko', 'ar', 'hi', 'ru', 
                              'pt', 'it', 'tr', 'th', 'vi', 'pl', 'uk', 'nl', 'id', 'sv']
        
        # Data source configuration
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
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.logger.info(f"📊 Initialized CuratedDataDownloader for {len(self.languages)} languages")
    
    def get_english_pairs(self) -> List[str]:
        """Return list of target languages for English-centric pairs"""
        # Filter out 'en' from languages
        return [lang for lang in self.languages if lang != 'en']
    
    def download_essential_data(self, output_dir: str = 'data/essential') -> Dict[str, int]:
        """
        Download essential, high-quality data with modern HuggingFace API
        
        Args:
            output_dir: Directory to save downloaded data
            
        Returns:
            Dictionary with download statistics
        """
        output_path = DirectoryManager.create_directory(output_dir)
        
        stats = {
            'flores200': 0,
            'tatoeba': 0,
            'opus_samples': 0,
            'total_files': 0,
            'total_size_mb': 0
        }
        
        # 1. FLORES-200 for evaluation
        self.logger.info("📥 Downloading FLORES-200...")
        flores_success = self._download_flores200(output_path)
        if flores_success:
            stats['flores200'] = 1
            stats['total_files'] += 1
        
        # 2. Tatoeba for diverse examples
        self.logger.info("📥 Downloading Tatoeba sentences...")
        tatoeba_count = self._download_tatoeba(output_path)
        stats['tatoeba'] = tatoeba_count
        stats['total_files'] += tatoeba_count
        
        # 3. OpenSubtitles sample
        self.logger.info("📥 Downloading OpenSubtitles sample...")
        opus_count = self._download_opus_samples(output_path)
        stats['opus_samples'] = opus_count
        stats['total_files'] += opus_count
        
        # Calculate total size
        for file_path in output_path.rglob('*'):
            if file_path.is_file():
                stats['total_size_mb'] += file_path.stat().st_size / (1024 * 1024)
        
        self.logger.info(f"✅ Downloaded {stats['total_files']} datasets ({stats['total_size_mb']:.1f}MB)")
        return stats
    
    def _download_flores200(self, output_dir: Path) -> bool:
        """Download FLORES-200 evaluation dataset"""
        try:
            if INTEGRATED_MODE:
                # Use shared dataset loader
                dataset = self.dataset_loader.load_dataset_safely(
                    self.data_sources['flores200']['dataset_name'],
                    config_name=self.data_sources['flores200']['config_name'],
                    split=self.data_sources['flores200']['split']
                )
                
                if dataset:
                    save_path = output_dir / 'flores200'
                    dataset.save_to_disk(str(save_path))
                    self.logger.info(f"✓ FLORES-200 saved to {save_path}")
                    return True
            else:
                # Fallback for standalone mode
                flores = load_dataset(
                    self.data_sources['flores200']['dataset_name'],
                    name=self.data_sources['flores200']['config_name'],
                    split=self.data_sources['flores200']['split'],
                    trust_remote_code=True
                )
                save_path = output_dir / 'flores200'
                flores.save_to_disk(str(save_path))
                self.logger.info(f"✓ FLORES-200 saved to {save_path}")
                return True
                
        except Exception as e:
            self.logger.error(f"✗ Failed to download FLORES-200: {e}")
            return False
    
    def _download_tatoeba(self, output_dir: Path) -> int:
        """Download Tatoeba sentence pairs"""
        downloaded_count = 0
        
        for target_lang in tqdm(self.get_english_pairs(), desc="Downloading Tatoeba"):
            try:
                if INTEGRATED_MODE:
                    # Use shared dataset loader
                    dataset = self.dataset_loader.load_dataset_safely(
                        self.data_sources['tatoeba']['dataset_name'],
                        lang1='eng',
                        lang2=target_lang,
                        split='train'
                    )
                    
                    if dataset:
                        # Limit to 100k sentences
                        if hasattr(dataset, '__len__') and len(dataset) > 100000:
                            dataset = dataset.select(range(100000))
                        
                        save_path = output_dir / f'tatoeba_en_{target_lang}'
                        dataset.save_to_disk(str(save_path))
                        self.logger.info(f"✓ en-{target_lang}: saved to {save_path}")
                        downloaded_count += 1
                else:
                    # Fallback for standalone mode
                    dataset = load_dataset(
                        self.data_sources['tatoeba']['dataset_name'],
                        lang1='eng',
                        lang2=target_lang,
                        split='train',
                        trust_remote_code=True
                    )
                    
                    # Limit to 100k sentences
                    dataset = dataset.select(range(min(100000, len(dataset))))
                    
                    save_path = output_dir / f'tatoeba_en_{target_lang}'
                    dataset.save_to_disk(str(save_path))
                    self.logger.info(f"✓ en-{target_lang}: {len(dataset)} sentences")
                    downloaded_count += 1
                    
            except Exception as e:
                self.logger.debug(f"✗ en-{target_lang}: {e}")
        
        return downloaded_count
    
    def _download_opus_samples(self, output_dir: Path) -> int:
        """Download OPUS corpus samples"""
        opus_corpora = ['OpenSubtitles', 'MultiUN']
        downloaded_count = 0
        
        for corpus_name in opus_corpora:
            success = self.download_opus_sample(
                corpus_name=corpus_name,
                output_dir=output_dir,
                max_size_mb=100
            )
            if success:
                downloaded_count += 1
        
        return downloaded_count
    
    def download_opus_sample(
        self, 
        corpus_name: str, 
        output_dir: Path, 
        max_size_mb: int = 100
    ) -> bool:
        """
        Download sample from OPUS using modern requests with streaming
        
        Args:
            corpus_name: Name of OPUS corpus (e.g., 'OpenSubtitles', 'MultiUN')
            output_dir: Output directory
            max_size_mb: Maximum download size in MB
            
        Returns:
            True if successful
        """
        base_url = f"https://object.pouta.csc.fi/OPUS-{corpus_name}/v2018/moses"
        opus_dir = DirectoryManager.create_directory(output_dir / 'opus')
        
        # Focus on major language pairs
        lang_pairs = ['en-es', 'en-fr', 'en-de', 'en-zh', 'en-ru', 'en-ja', 'en-ar']
        success_count = 0
        
        for lang_pair in lang_pairs:
            try:
                # Construct URL
                url = f"{base_url}/{lang_pair}.txt.zip"
                output_file = opus_dir / f'{corpus_name}_{lang_pair}.zip'
                
                self.logger.info(f"📥 Downloading {corpus_name} {lang_pair}...")
                
                # Download with size limit
                size_downloaded = 0
                chunk_size = 1024 * 1024  # 1MB chunks
                
                with self.session.get(url, stream=True, timeout=30) as response:
                    response.raise_for_status()
                    
                    # Get total size if available
                    total_size = int(response.headers.get('content-length', 0))
                    if total_size > 0:
                        self.logger.info(f"  Total size: {total_size / (1024*1024):.1f}MB")
                    
                    # Download with progress bar
                    with open(output_file, 'wb') as f:
                        with tqdm(
                            total=min(total_size, max_size_mb * 1024 * 1024),
                            unit='B',
                            unit_scale=True,
                            desc=f"{lang_pair}"
                        ) as pbar:
                            for chunk in response.iter_content(chunk_size=chunk_size):
                                if chunk:
                                    f.write(chunk)
                                    size_downloaded += len(chunk)
                                    pbar.update(len(chunk))
                                    
                                    # Stop if size limit reached
                                    if size_downloaded >= max_size_mb * 1024 * 1024:
                                        self.logger.info(f"  Size limit reached ({max_size_mb}MB)")
                                        break
                
                # Extract if successful
                if output_file.exists():
                    self._extract_opus_file(output_file, opus_dir)
                    success_count += 1
                    self.logger.info(f"✓ {lang_pair}: {size_downloaded / (1024*1024):.1f}MB")
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    self.logger.debug(f"✗ {lang_pair}: Not available")
                else:
                    self.logger.error(f"✗ {lang_pair}: HTTP {e.response.status_code}")
            except Exception as e:
                self.logger.error(f"✗ {lang_pair}: {e}")
        
        return success_count > 0
    
    def _extract_opus_file(self, zip_file: Path, output_dir: Path) -> None:
        """Extract OPUS zip file and convert to tab-separated format"""
        try:
            with zipfile.ZipFile(zip_file, 'r') as z:
                # Get language pair from filename
                lang_pair = zip_file.stem.split('_')[-1]  # e.g., 'en-es'
                
                # Look for the parallel files
                files = z.namelist()
                source_file = None
                target_file = None
                
                for f in files:
                    if f.endswith(f'.{lang_pair.split("-")[0]}'):
                        source_file = f
                    elif f.endswith(f'.{lang_pair.split("-")[1]}'):
                        target_file = f
                
                if source_file and target_file:
                    # Read both files
                    with z.open(source_file) as sf, z.open(target_file) as tf:
                        source_lines = sf.read().decode('utf-8').splitlines()
                        target_lines = tf.read().decode('utf-8').splitlines()
                    
                    # Write tab-separated file
                    output_file = output_dir / f"{zip_file.stem}.txt"
                    with open(output_file, 'w', encoding='utf-8') as out:
                        for src, tgt in zip(source_lines, target_lines):
                            if src.strip() and tgt.strip():
                                out.write(f"{src.strip()}\t{tgt.strip()}\n")
                    
                    # Remove zip file to save space
                    zip_file.unlink()
                    
                    self.logger.info(f"  Extracted {len(source_lines)} sentence pairs")
                else:
                    self.logger.warning(f"  Could not find parallel files in {zip_file.name}")
                    
        except Exception as e:
            self.logger.error(f"  Failed to extract {zip_file.name}: {e}")
    
    def download_specific_dataset(
        self, 
        dataset_key: str, 
        output_dir: str = 'data/essential'
    ) -> bool:
        """
        Download a specific dataset by key (production).
        """
        import requests
        import tarfile
        import os
        if dataset_key not in self.data_sources:
            self.logger.error(f"Unknown dataset: {dataset_key}")
            return False
        output_path = DirectoryManager.create_directory(output_dir)
        if dataset_key == 'flores200':
            return self._download_flores200(output_path)
        elif dataset_key == 'tatoeba':
            return self._download_tatoeba(output_path) > 0
        elif dataset_key == 'opus_books':
            url = 'https://object.pouta.csc.fi/OPUS-Books/v1/moses/en-es.txt.zip'
            local_zip = os.path.join(output_path, 'opus_books_en-es.txt.zip')
            r = requests.get(url, stream=True)
            with open(local_zip, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            # Extract if needed
            # ... (add extraction logic)
            return os.path.exists(local_zip)
        elif dataset_key == 'ted_talks':
            url = 'https://opus.nlpl.eu/download.php?f=TED2020/v1/moses/en-es.txt.zip'
            local_zip = os.path.join(output_path, 'ted_talks_en-es.txt.zip')
            r = requests.get(url, stream=True)
            with open(local_zip, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            # Extract if needed
            # ... (add extraction logic)
            return os.path.exists(local_zip)
        return False
    
    def get_dataset_info(self) -> Dict[str, Dict]:
        """Get information about available datasets"""
        return self.data_sources.copy()


def main():
    """Main entry point for standalone execution"""
    downloader = CuratedDataDownloader()
    
    # Show available datasets
    print("📊 Available Curated Datasets:")
    print("-" * 60)
    for name, info in downloader.get_dataset_info().items():
        print(f"\n{name}:")
        for key, value in info.items():
            if key != 'dataset_name':
                print(f"  {key}: {value}")
    
    # Download all essential data
    print("\n" + "="*60)
    print("📥 Starting download of essential data...")
    print("="*60)
    
    stats = downloader.download_essential_data()
    
    # Print summary
    print("\n" + "="*60)
    print("📊 DOWNLOAD SUMMARY:")
    print("="*60)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.1f}")
        else:
            print(f"{key}: {value}")
    
    print("\n✅ Essential data download complete!")


if __name__ == "__main__":
    main()