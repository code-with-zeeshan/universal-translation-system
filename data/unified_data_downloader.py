# data/unified_data_downloader.py
"""
Unified data downloader combining curated, training, and smart strategies
Replaces: download_curated_data.py, download_training_data.py, smart_data_downloader.py
"""

import logging
import zipfile
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from datasets import load_dataset

from config.schemas import RootConfig
from utils.exceptions import DataError
from utils.common_utils import DirectoryManager
from utils.security import validate_model_source
from .data_utils import DataProcessor, DatasetLoader

class DatasetType(Enum):
    """Types of datasets for different purposes"""
    EVALUATION = "evaluation"    # High-quality, small (FLORES, Tatoeba)
    TRAINING = "training"        # Large-scale training data
    SAMPLE = "sample"           # Limited samples for testing

class DownloadPriority(Enum):
    """Download priority levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class LanguagePair:
    """Language pair with metadata"""
    source: str
    target: str
    priority: DownloadPriority
    expected_size: int
    data_sources: List[str]
    dataset_type: DatasetType = DatasetType.TRAINING
    
    @property
    def pair_string(self) -> str:
        return f"{self.source}-{self.target}"
    
    def to_dict(self) -> dict:
        return {
            'pair': self.pair_string,
            'priority': self.priority.value,
            'expected_size': self.expected_size,
            'sources': self.data_sources,
            'type': self.dataset_type.value
        }

class UnifiedDataDownloader:
    """
    Unified data downloader combining all download functionality.
    Replaces CuratedDataDownloader, MultilingualDataCollector, and SmartDataStrategy.
    """
    
    def __init__(self, config: RootConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize utilities
        self.data_processor = DataProcessor(config, self.logger)
        self.dataset_loader = DatasetLoader(self.logger)
        
        # Configuration
        self.languages = config.data.active_languages
        self.training_distribution = config.data.training_distribution
        
        # Setup HTTP session with retries (from curated downloader)
        self.session = self._setup_http_session()
        
        # Data sources (combined from all three)
        self.data_sources = self._initialize_data_sources()
        
        # Strategy configuration (from smart downloader)
        self.priority_rules = config.data_strategy.priority_rules if hasattr(config, 'data_strategy') else {}
        self.source_preferences = config.data_strategy.source_preferences if hasattr(config, 'data_strategy') else {}
        
        self.logger.info(f"📊 UnifiedDataDownloader initialized for {len(self.languages)} languages")
    
    def _setup_http_session(self) -> requests.Session:
        """Setup HTTP session with retry strategy"""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    def _initialize_data_sources(self) -> Dict[str, Dict]:
        """Initialize all data sources from all three downloaders"""
        return {
            # From curated downloader
            'flores200': {
                'dataset_name': 'facebook/flores',
                'config_name': 'flores200_sacrebleu_tokenized_xlm_roberta_base',
                'split': 'dev',
                'type': DatasetType.EVALUATION,
                'size': '~10MB per language',
                'sentences': 1000,
                'quality': 'excellent'
            },
            'tatoeba': {
                'dataset_name': 'Helsinki-NLP/tatoeba',
                'type': DatasetType.EVALUATION,
                'size': '~5-50MB per pair',
                'sentences': '1k-500k',
                'quality': 'good'
            },
            
            # From training downloader
            'opus-100': {
                'dataset_name': 'Helsinki-NLP/opus-100',
                'type': DatasetType.TRAINING,
                'streaming': True,
                'quality': 'good'
            },
            'opus_opensubtitles': {
                'dataset_name': 'Helsinki-NLP/opus_opensubtitles',
                'type': DatasetType.TRAINING,
                'streaming': True,
                'quality': 'moderate'
            },
            'nllb-seed': {
                'dataset_name': 'facebook/nllb-seed',
                'type': DatasetType.TRAINING,
                'streaming': True,
                'quality': 'excellent'
            },
            'ccmatrix': {
                'dataset_name': 'yhavinga/ccmatrix',
                'config_name': 'multilingual',
                'type': DatasetType.TRAINING,
                'streaming': True,
                'quality': 'good'
            },
            
            # WMT datasets
            'wmt19': {'dataset_name': 'wmt19', 'type': DatasetType.TRAINING},
            'wmt20': {'dataset_name': 'wmt20', 'type': DatasetType.TRAINING},
            'wmt21': {'dataset_name': 'wmt21', 'type': DatasetType.TRAINING},
        }
    
    # ============= SMART STRATEGY METHODS (from smart_data_downloader.py) =============
    
    def get_required_pairs(self, dataset_type: DatasetType = DatasetType.TRAINING) -> List[LanguagePair]:
        """Get strategically selected language pairs"""
        required_pairs = []
        
        if dataset_type == DatasetType.EVALUATION:
            # For evaluation, just need English pairs
            for lang in self.languages:
                if lang != 'en':
                    required_pairs.append(LanguagePair(
                        source='en',
                        target=lang,
                        priority=DownloadPriority.HIGH,
                        expected_size=1000,  # FLORES size
                        data_sources=['flores200', 'tatoeba'],
                        dataset_type=DatasetType.EVALUATION
                    ))
        else:
            # Training pairs from configuration
            for pair_str, expected_size in self.training_distribution.items():
                if '-' in pair_str:
                    source, target = pair_str.split('-')
                    priority = self._determine_priority(pair_str)
                    sources = self._get_data_sources(source, target)
                    
                    required_pairs.append(LanguagePair(
                        source=source,
                        target=target,
                        priority=priority,
                        expected_size=expected_size,
                        data_sources=sources,
                        dataset_type=dataset_type
                    ))
        
        # Sort by priority
        priority_order = {DownloadPriority.HIGH: 0, DownloadPriority.MEDIUM: 1, DownloadPriority.LOW: 2}
        required_pairs.sort(key=lambda p: (priority_order[p.priority], -p.expected_size))
        
        return required_pairs
    
    def _determine_priority(self, pair_str: str) -> DownloadPriority:
        """Determine priority of a language pair"""
        # Check configured rules first
        for priority, pairs in self.priority_rules.items():
            if pair_str in pairs:
                return DownloadPriority(priority)
        
        # Default rules
        source, target = pair_str.split('-')
        if source == 'en' and target in ['es', 'fr', 'de', 'zh', 'ru']:
            return DownloadPriority.HIGH
        elif source == 'en' or target == 'en':
            return DownloadPriority.MEDIUM
        else:
            return DownloadPriority.LOW
    
    def _get_data_sources(self, source: str, target: str) -> List[str]:
        """Get recommended data sources for a language pair"""
        # Use preferences if configured
        if self.source_preferences:
            if source == 'en' or target == 'en':
                return self.source_preferences.get('en_centric', ['opus-100', 'nllb-seed'])
            
            european = ['es', 'fr', 'de', 'it', 'pt', 'nl', 'sv', 'pl']
            if source in european and target in european:
                return self.source_preferences.get('european', ['opus-100'])
            
            asian = ['zh', 'ja', 'ko', 'th', 'vi', 'id']
            if source in asian and target in asian:
                return self.source_preferences.get('asian', ['ccmatrix'])
        
        # Default sources
        return ['opus-100', 'nllb-seed', 'ccmatrix']
    
    def get_download_schedule(self, dataset_type: DatasetType = DatasetType.TRAINING) -> List[Dict]:
        """Get optimized download schedule with parallel batches"""
        pairs = self.get_required_pairs(dataset_type)
        
        schedule = []
        
        if dataset_type == DatasetType.EVALUATION:
            # Simple schedule for evaluation data
            schedule.append({
                'batch_name': 'Evaluation Datasets',
                'pairs': pairs,
                'parallel': True,
                'max_workers': 4
            })
        else:
            # Smart batching for training data
            # High priority English pairs
            batch1 = [p for p in pairs if p.priority == DownloadPriority.HIGH and p.source == 'en']
            if batch1:
                schedule.append({
                    'batch_name': 'High Priority English Pairs',
                    'pairs': batch1,
                    'parallel': True,
                    'max_workers': 4
                })
            
            # Direct pairs (non-English)
            batch2 = [p for p in pairs if p.source != 'en' and p.target != 'en']
            if batch2:
                schedule.append({
                    'batch_name': 'Direct Language Pairs',
                    'pairs': batch2,
                    'parallel': True,
                    'max_workers': 2
                })
            
            # Remaining pairs
            batch3 = [p for p in pairs if p not in batch1 and p not in batch2]
            if batch3:
                schedule.append({
                    'batch_name': 'Additional Pairs',
                    'pairs': batch3,
                    'parallel': True,
                    'max_workers': 3
                })
        
        return schedule
    
    # ============= DOWNLOAD METHODS (combined from all three) =============
    
    def download_all(self, 
                    output_dir: str = 'data',
                    dataset_types: List[DatasetType] = None) -> Dict[str, Any]:
        """
        Main download method - replaces all individual download methods
        
        Args:
            output_dir: Base output directory
            dataset_types: Types of datasets to download (None = all)
        
        Returns:
            Statistics dictionary
        """
        if dataset_types is None:
            dataset_types = [DatasetType.EVALUATION, DatasetType.TRAINING]
        
        stats = {'total_files': 0, 'total_size_mb': 0}
        
        for dtype in dataset_types:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"📥 Downloading {dtype.value} datasets...")
            self.logger.info(f"{'='*60}")
            
            if dtype == DatasetType.EVALUATION:
                eval_stats = self._download_evaluation_data(output_dir)
                stats.update(eval_stats)
            elif dtype == DatasetType.TRAINING:
                train_stats = self._download_training_data(output_dir)
                stats.update(train_stats)
        
        return stats
    
    def _download_evaluation_data(self, base_dir: str) -> Dict[str, Any]:
        """Download evaluation datasets (from curated downloader)"""
        output_dir = Path(base_dir) / 'evaluation'
        DirectoryManager.create_directory(output_dir)
        
        stats = {}
        
        # FLORES-200
        if self._download_flores200(output_dir):
            stats['flores200'] = 1
        
        # Tatoeba
        tatoeba_count = self._download_tatoeba(output_dir)
        stats['tatoeba'] = tatoeba_count
        
        # OPUS samples
        opus_count = self._download_opus_samples(output_dir)
        stats['opus_samples'] = opus_count
        
        return stats
    
    def _download_training_data(self, base_dir: str) -> Dict[str, Any]:
        """Download training datasets with smart strategy"""
        output_dir = Path(base_dir) / 'training'
        DirectoryManager.create_directory(output_dir)
        
        stats = {'downloaded_pairs': 0}
        
        # Get download schedule
        schedule = self.get_download_schedule(DatasetType.TRAINING)
        
        for batch in schedule:
            self.logger.info(f"\n📦 {batch['batch_name']}:")
            self.logger.info(f"  Pairs: {len(batch['pairs'])}")
            self.logger.info(f"  Parallel: {batch['parallel']}")
            
            if batch['parallel']:
                # Parallel download
                with ThreadPoolExecutor(max_workers=batch['max_workers']) as executor:
                    futures = []
                    for pair in batch['pairs']:
                        future = executor.submit(
                            self._download_pair_data,
                            pair,
                            output_dir
                        )
                        futures.append((pair, future))
                    
                    # Collect results
                    for pair, future in tqdm(futures, desc=batch['batch_name']):
                        try:
                            if future.result():
                                stats['downloaded_pairs'] += 1
                        except Exception as e:
                            self.logger.error(f"Failed to download {pair.pair_string}: {e}")
            else:
                # Sequential download
                for pair in tqdm(batch['pairs'], desc=batch['batch_name']):
                    if self._download_pair_data(pair, output_dir):
                        stats['downloaded_pairs'] += 1
        
        return stats
    
    def _download_pair_data(self, pair: LanguagePair, output_dir: Path) -> bool:
        """Download data for a specific language pair"""
        success = False
        pair_dir = output_dir / pair.pair_string
        DirectoryManager.create_directory(pair_dir)
        
        for source_name in pair.data_sources:
            if source_name not in self.data_sources:
                continue
            
            source_info = self.data_sources[source_name]
            
            try:
                # Validate source
                if not validate_model_source(source_info['dataset_name']):
                    self.logger.warning(f"Skipping untrusted source: {source_name}")
                    continue
                
                # Download based on source type
                if source_info['dataset_name'].startswith('Helsinki-NLP/opus'):
                    success = self._download_opus_pair(pair, source_info, pair_dir)
                elif 'streaming' in source_info and source_info['streaming']:
                    success = self._download_streaming_dataset(pair, source_info, pair_dir)
                else:
                    success = self._download_standard_dataset(pair, source_info, pair_dir)
                
                if success:
                    break  # Stop after first successful download
                    
            except Exception as e:
                self.logger.error(f"Error downloading {source_name} for {pair.pair_string}: {e}")
        
        return success
    
    def _download_flores200(self, output_dir: Path) -> bool:
        """Download FLORES-200 evaluation dataset"""
        try:
            dataset = self.dataset_loader.load_dataset_safely(
                'facebook/flores',
                config_name='flores200_sacrebleu_tokenized_xlm_roberta_base',
                split='dev',
                trust_remote_code=False
            )
            
            if dataset:
                save_path = output_dir / 'flores200'
                dataset.save_to_disk(str(save_path))
                self.logger.info(f"✓ FLORES-200 saved to {save_path}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to download FLORES-200: {e}")
        return False
    
    def _download_tatoeba(self, output_dir: Path) -> int:
        """Download Tatoeba datasets"""
        count = 0
        for target_lang in self.languages:
            if target_lang == 'en':
                continue
            
            try:
                dataset = self.dataset_loader.load_dataset_safely(
                    'Helsinki-NLP/tatoeba',
                    lang1='eng',
                    lang2=target_lang,
                    split='train',
                    trust_remote_code=False
                )
                
                if dataset:
                    # Limit size
                    if len(dataset) > 100000:
                        dataset = dataset.select(range(100000))
                    
                    save_path = output_dir / f'tatoeba_en_{target_lang}'
                    dataset.save_to_disk(str(save_path))
                    count += 1
            except Exception as e:
                self.logger.debug(f"No Tatoeba data for en-{target_lang}: {e}")
        
        return count
    
    def _download_opus_samples(self, output_dir: Path) -> int:
        """Download OPUS samples with size limits"""
        opus_dir = output_dir / 'opus_samples'
        DirectoryManager.create_directory(opus_dir)
        
        count = 0
        corpora = ['OpenSubtitles', 'MultiUN']
        lang_pairs = ['en-es', 'en-fr', 'en-de', 'en-zh', 'en-ru']
        
        for corpus in corpora:
            for lang_pair in lang_pairs:
                if self._download_opus_file(corpus, lang_pair, opus_dir, max_size_mb=50):
                    count += 1
        
        return count
    
    def _download_opus_file(self, 
                           corpus: str, 
                           lang_pair: str, 
                           output_dir: Path,
                           max_size_mb: int = 100) -> bool:
        """Download and extract OPUS file with size limit"""
        url = f"https://object.pouta.csc.fi/OPUS-{corpus}/v2018/moses/{lang_pair}.txt.zip"
        output_file = output_dir / f'{corpus}_{lang_pair}.zip'
        
        try:
            # Download with size limit
            size_downloaded = 0
            chunk_size = 1024 * 1024
            
            with self.session.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()
                
                with open(output_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            size_downloaded += len(chunk)
                            
                            if size_downloaded >= max_size_mb * 1024 * 1024:
                                break
            
            # Extract
            if output_file.exists():
                self._extract_opus_file(output_file, output_dir)
                return True
                
        except Exception as e:
            self.logger.debug(f"Failed to download {corpus} {lang_pair}: {e}")
        
        return False
    
    def _extract_opus_file(self, zip_file: Path, output_dir: Path) -> None:
        """Extract OPUS zip file to TSV format"""
        try:
            with zipfile.ZipFile(zip_file, 'r') as z:
                lang_pair = zip_file.stem.split('_')[-1]
                src_lang, tgt_lang = lang_pair.split('-')
                
                # Find parallel files
                files = z.namelist()
                src_file = next((f for f in files if f.endswith(f'.{src_lang}')), None)
                tgt_file = next((f for f in files if f.endswith(f'.{tgt_lang}')), None)
                
                if src_file and tgt_file:
                    # Read and combine
                    with z.open(src_file) as sf, z.open(tgt_file) as tf:
                        src_lines = sf.read().decode('utf-8').splitlines()
                        tgt_lines = tf.read().decode('utf-8').splitlines()
                    
                    # Write TSV
                    tsv_file = output_dir / f"{zip_file.stem}.tsv"
                    with open(tsv_file, 'w', encoding='utf-8') as out:
                        for src, tgt in zip(src_lines, tgt_lines):
                            if src.strip() and tgt.strip():
                                out.write(f"{src.strip()}\t{tgt.strip()}\n")
                    
                    # Clean up
                    zip_file.unlink()
                    self.logger.debug(f"Extracted {len(src_lines)} pairs from {zip_file.name}")
                    
        except Exception as e:
            self.logger.error(f"Extraction failed for {zip_file.name}: {e}")
    
    def _download_streaming_dataset(self, 
                                   pair: LanguagePair,
                                   source_info: Dict,
                                   output_dir: Path) -> bool:
        """Download dataset with streaming support"""
        try:
            dataset = self.dataset_loader.load_dataset_safely(
                source_info['dataset_name'],
                config_name=source_info.get('config_name', pair.pair_string),
                streaming=True,
                trust_remote_code=False
            )
            
            if dataset:
                self.data_processor.process_streaming_dataset(
                    dataset,
                    output_dir / source_info['dataset_name'].split('/')[-1],
                    max_samples=pair.expected_size
                )
                return True
        except Exception as e:
            self.logger.debug(f"Streaming download failed: {e}")
        return False
    
    def _download_standard_dataset(self,
                                  pair: LanguagePair,
                                  source_info: Dict,
                                  output_dir: Path) -> bool:
        """Download standard dataset"""
        try:
            dataset = self.dataset_loader.load_dataset_safely(
                source_info['dataset_name'],
                config_name=source_info.get('config_name'),
                split=source_info.get('split', 'train'),
                trust_remote_code=False
            )
            
            if dataset:
                save_path = output_dir / source_info['dataset_name'].split('/')[-1]
                dataset.save_to_disk(str(save_path))
                return True
        except Exception as e:
            self.logger.debug(f"Standard download failed: {e}")
        return False
    
    def _download_opus_pair(self,
                           pair: LanguagePair,
                           source_info: Dict,
                           output_dir: Path) -> bool:
        """Download OPUS dataset for language pair"""
        corpus = source_info['dataset_name'].split('/')[-1].replace('opus_', '')
        return self._download_opus_file(
            corpus.capitalize(),
            pair.pair_string,
            output_dir,
            max_size_mb=500 if pair.priority == DownloadPriority.HIGH else 100
        )
    
    # ============= UTILITY METHODS =============
    
    def estimate_download_size(self, dataset_types: List[DatasetType] = None) -> Dict[str, float]:
        """Estimate total download size"""
        if dataset_types is None:
            dataset_types = [DatasetType.EVALUATION, DatasetType.TRAINING]
        
        estimates = {'total_gb': 0}
        
        for dtype in dataset_types:
            pairs = self.get_required_pairs(dtype)
            
            if dtype == DatasetType.EVALUATION:
                # Small evaluation datasets
                estimates['evaluation_gb'] = len(pairs) * 0.01  # ~10MB per pair
            else:
                # Training data estimates
                bytes_per_sentence = {
                    DownloadPriority.HIGH: 150,
                    DownloadPriority.MEDIUM: 120,
                    DownloadPriority.LOW: 100
                }
                
                size_gb = sum(
                    (p.expected_size * bytes_per_sentence[p.priority]) / (1024**3)
                    for p in pairs
                )
                estimates['training_gb'] = size_gb
            
            estimates['total_gb'] += estimates.get(f'{dtype.value}_gb', 0)
        
        return estimates
    
    def export_strategy(self, output_file: str = 'data/download_strategy.json') -> None:
        """Export download strategy to JSON"""
        strategy = {
            'evaluation': {
                'pairs': [p.to_dict() for p in self.get_required_pairs(DatasetType.EVALUATION)],
                'schedule': self.get_download_schedule(DatasetType.EVALUATION)
            },
            'training': {
                'pairs': [p.to_dict() for p in self.get_required_pairs(DatasetType.TRAINING)],
                'schedule': self.get_download_schedule(DatasetType.TRAINING)
            },
            'estimates': self.estimate_download_size()
        }
        
        with open(output_file, 'w') as f:
            json.dump(strategy, f, indent=2)
        
        self.logger.info(f"📄 Strategy exported to {output_file}")
    
    def download_specific_dataset(self, 
                                 dataset_key: str,
                                 output_dir: str = 'data') -> bool:
        """Download a specific dataset by key (backward compatibility)"""
        if dataset_key not in self.data_sources:
            self.logger.error(f"Unknown dataset: {dataset_key}")
            return False
        
        source_info = self.data_sources[dataset_key]
        dtype = source_info.get('type', DatasetType.TRAINING)
        
        output_path = Path(output_dir) / dtype.value / dataset_key
        DirectoryManager.create_directory(output_path)
        
        # Create a dummy pair for the download
        pair = LanguagePair(
            source='en',
            target='es',  # Default pair
            priority=DownloadPriority.HIGH,
            expected_size=100000,
            data_sources=[dataset_key],
            dataset_type=dtype
        )
        
        return self._download_pair_data(pair, output_path.parent)