# data/unified_data_downloader.py
"""
Unified data downloader combining curated, training, and smart strategies
Replaces: download_curated_data.py, download_training_data.py, smart_data_downloader.py
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
import zipfile
import io
# Optional dependency: requests. Provide minimal shim for smoke/dry-run.
try:
    import requests  # type: ignore
    from requests.adapters import HTTPAdapter  # type: ignore
    from urllib3.util.retry import Retry  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore
    HTTPAdapter = None  # type: ignore
    class Retry:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
from tqdm import tqdm
# Optional dependency; for smoke/dry-run we won't load real datasets
try:
    from datasets import load_dataset  # type: ignore
except Exception:  # pragma: no cover
    load_dataset = None  # type: ignore

from config.schemas import RootConfig
from utils.exceptions import DataError
from utils.common_utils import DirectoryManager
from utils.security import validate_model_source
from data.data_utils import DataProcessor, DatasetLoader

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
        # Support both object-attr and dict-style configs
        data_cfg = getattr(config, 'data', {})
        if isinstance(data_cfg, dict):
            self.languages = data_cfg.get('active_languages', ['en'])
            self.training_distribution = data_cfg.get('training_distribution', {})
        else:
            self.languages = getattr(data_cfg, 'active_languages', ['en'])
            self.training_distribution = getattr(data_cfg, 'training_distribution', {})
        
        # Setup HTTP session with retries (from curated downloader)
        self.session = self._setup_http_session()
        
        # Data sources (combined from all three)
        self.data_sources = self._initialize_data_sources()
        
        # Strategy configuration (from smart downloader)
        ds_cfg = getattr(config, 'data_strategy', {})
        if isinstance(ds_cfg, dict):
            self.priority_rules = ds_cfg.get('priority_rules', {})
            self.source_preferences = ds_cfg.get('source_preferences', {})
        else:
            self.priority_rules = getattr(ds_cfg, 'priority_rules', {})
            self.source_preferences = getattr(ds_cfg, 'source_preferences', {})
        
        self.logger.info(f"📊 UnifiedDataDownloader initialized for {len(self.languages)} languages")
    
    def _setup_http_session(self):
        """Setup HTTP session with retry strategy. Returns None if requests is unavailable."""
        if requests is None or HTTPAdapter is None:
            self.logger.warning("'requests' not installed; HTTP session disabled (smoke mode)")
            return None
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
    
    _OPUS_DIRECT_URL = "https://opus.nlpl.eu/download.php?f={corpus}/{lang_pair}.txt.zip"

    def _initialize_data_sources(self) -> Dict[str, Dict]:
        """Initialize all data sources (tried in order per pair)"""
        return {
            'opus-100': {
                'dataset_name': 'Helsinki-NLP/opus-100',
                'type': DatasetType.TRAINING,
                'streaming': True,
                'quality': 'good'
            },
            'opus_paracrawl': {
                'dataset_name': 'Helsinki-NLP/opus_paracrawl',
                'type': DatasetType.TRAINING,
                'streaming': True,
                'quality': 'medium',
            },
            'opus_ccmatrix': {
                'dataset_name': 'Helsinki-NLP/opus_ccmatrix',
                'type': DatasetType.TRAINING,
                'streaming': True,
                'quality': 'medium',
            },
            'open_subtitles': {
                'dataset_name': 'Helsinki-NLP/open_subtitles',
                'type': DatasetType.TRAINING,
                'streaming': True,
                'quality': 'medium',
            },
            'opus_wikimatrix': {
                'dataset_name': 'Helsinki-NLP/opus_wikimatrix',
                'type': DatasetType.TRAINING,
                'streaming': True,
                'quality': 'medium',
            },
            'opus_direct': {
                'type': DatasetType.TRAINING,
                'streaming': False,
                'quality': 'medium',
            },
            'wmt19': {'dataset_name': 'wmt19', 'type': DatasetType.TRAINING},
            'wmt20': {'dataset_name': 'wmt20', 'type': DatasetType.TRAINING},
            'wmt21': {'dataset_name': 'wmt21', 'type': DatasetType.TRAINING},
        }
    
    # ============= SMART STRATEGY METHODS (from smart_data_downloader.py) =============
    
    def get_required_pairs(self, dataset_type: DatasetType = DatasetType.TRAINING) -> List[LanguagePair]:
        """Get strategically selected language pairs"""
        required_pairs = []
        
        if dataset_type == DatasetType.EVALUATION:
            for lang in self.languages:
                if lang != 'en':
                    required_pairs.append(LanguagePair(
                        source='en',
                        target=lang,
                        priority=DownloadPriority.HIGH,
                        expected_size=1000,
                        data_sources=['opus-100'],
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
        """Get recommended data sources for a language pair (tried in order)"""
        if self.source_preferences:
            if source == 'en' or target == 'en':
                return self.source_preferences.get('en_centric', ['opus-100'])
            european = ['es', 'fr', 'de', 'it', 'pt', 'nl', 'sv', 'pl']
            if source in european and target in european:
                return self.source_preferences.get('european', ['opus-100'])
            asian = ['zh', 'ja', 'ko', 'th', 'vi', 'id']
            if source in asian and target in asian:
                return self.source_preferences.get('asian', ['opus-100'])
        return ['opus-100']
    
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
        """Download evaluation datasets from opus-100 test splits"""
        output_dir = Path(base_dir) / 'evaluation'
        DirectoryManager.create_directory(output_dir)

        if load_dataset is None:
            self.logger.warning("datasets library not installed — skipping evaluation download")
            return {'eval_files': 0, 'eval_pairs': 0}

        stats = {'eval_files': 0, 'eval_pairs': 0}
        eval_pairs = self.get_required_pairs(DatasetType.EVALUATION)

        for pair in eval_pairs:
            pair_str = pair.pair_string
            output_file = output_dir / f"{pair_str}.tsv"

            if output_file.exists():
                self.logger.info(f"⏭️ {pair_str}.tsv already exists, skipping")
                stats['eval_files'] += 1
                stats['eval_pairs'] += 1
                continue

            # Try direct config first (en-X), then reversed (X-en) if that fails
            config_name = pair_str
            reverse_config = f"{pair.target}-{pair.source}"
            swap_src_tgt = False

            for attempt_cfg in [config_name, reverse_config]:
                self.logger.info(f"📥 Downloading opus-100 test split for {pair_str} (config={attempt_cfg})...")
                try:
                    dataset = self.dataset_loader.load_dataset_safely(
                        'Helsinki-NLP/opus-100',
                        config_name=attempt_cfg,
                        split='test',
                        streaming=True,
                    )

                    if dataset is None:
                        self.logger.warning(f"  No test split found for {attempt_cfg}")
                        continue

                    count = 0
                    with open(output_file, 'w', encoding='utf-8') as f:
                        for example in dataset:
                            translation = example.get('translation', {})
                            if attempt_cfg == reverse_config:
                                # Config is reversed, swap source/target extraction
                                src_text = translation.get(pair.target, '')
                                tgt_text = translation.get(pair.source, '')
                            else:
                                src_text = translation.get(pair.source, '')
                                tgt_text = translation.get(pair.target, '')
                            if src_text and tgt_text:
                                f.write(f"{src_text}\t{tgt_text}\t{pair.source}\t{pair.target}\n")
                                count += 1

                    if count > 0:
                        self.logger.info(f"✅ Downloaded {count:,} test samples for {pair_str}")
                        stats['eval_files'] += 1
                        stats['eval_pairs'] += 1
                        break
                    else:
                        self.logger.warning(f"  No samples for {attempt_cfg}, trying next")

                except Exception as e:
                    self.logger.warning(f"⚠️ Config {attempt_cfg} failed: {e}")
                    continue
            else:
                # Both config attempts failed
                self.logger.warning(f"  Failed to download eval data for {pair_str} (tried both {config_name} and {reverse_config})")
                output_file.unlink(missing_ok=True)
                continue

        self.logger.info(f"✅ Evaluation data: {stats['eval_files']} files, {stats['eval_pairs']} pairs")
        return stats
    
    def _download_training_data(self, base_dir: str) -> Dict[str, Any]:
        """Download training datasets with smart strategy"""
        output_dir = Path(base_dir) / 'raw'
        DirectoryManager.create_directory(output_dir)
        
        stats = {'downloaded_pairs': 0}

        # If running without requests/datasets, skip actual downloads in smoke mode
        if requests is None or load_dataset is None:
            self.logger.warning("Skipping training data download (requests/datasets not installed) - smoke mode")
            return stats
        
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
                # Special handling for direct OPUS download (no HF dataset)
                if source_name == 'opus_direct':
                    success = self._download_direct_opus(pair, output_dir)
                    if success:
                        break
                    continue

                # Validate source
                if not validate_model_source(source_info['dataset_name']):
                    self.logger.warning(f"Skipping untrusted source: {source_name}")
                    continue
                
                # Download based on source type
                if 'streaming' in source_info and source_info['streaming']:
                    success = self._download_streaming_dataset(pair, source_info, pair_dir)
                else:
                    success = self._download_standard_dataset(pair, source_info, pair_dir)
                
                if success:
                    break  # Stop after first successful download
                    
            except Exception as e:
                self.logger.error(f"Error downloading {source_name} for {pair.pair_string}: {e}")
        
        return success
    
    # Evaluation data is sourced from opus-100 test/validation splits
    # via the training pipeline. No separate evaluation download needed.
    
    def _download_streaming_dataset(self, 
                                   pair: LanguagePair,
                                   source_info: Dict,
                                   output_dir: Path) -> bool:
        """Download dataset with streaming support (falls back to non-streaming)"""
        for attempt_streaming in [True, False]:
            try:
                raw = self.dataset_loader.load_dataset_safely(
                    source_info['dataset_name'],
                    config_name=source_info.get('config_name', pair.pair_string),
                    streaming=attempt_streaming,
                )
                # load_dataset returns a DatasetDict when no split is given;
                # extract the 'train' split for training data.
                if hasattr(raw, 'keys'):
                    dataset = raw.get('train')
                    if dataset is None:
                        # fall back to the first available split
                        dataset = list(raw.values())[0]
                else:
                    dataset = raw
                if dataset is None:
                    continue
                # Write flat tab-separated file in the raw directory
                flat_file = output_dir.parent / f"{pair.pair_string}.txt"
                count = self.data_processor.process_streaming_dataset(
                    dataset,
                    flat_file,
                    max_samples=pair.expected_size,
                    source_lang=pair.source,
                    target_lang=pair.target,
                )
                if count > 0:
                    self.logger.info(f"✓ Downloaded {count:,} samples from {source_info['dataset_name']}")
                    return True
                self.logger.warning(f"0 samples from {source_info['dataset_name']}, trying next source")
            except Exception as e:
                mode = "streaming" if attempt_streaming else "non-streaming"
                self.logger.warning(f"{source_info['dataset_name']} ({mode}) failed: {e}")
                continue
        return False
    
    def _download_standard_dataset(self,
                                  pair: LanguagePair,
                                  source_info: Dict,
                                  output_dir: Path) -> bool:
        """Download standard (non-streaming) dataset"""
        try:
            dataset = self.dataset_loader.load_dataset_safely(
                source_info['dataset_name'],
                config_name=source_info.get('config_name'),
                split=source_info.get('split', 'train'),
            )
            if dataset:
                save_path = output_dir / source_info['dataset_name'].split('/')[-1]
                dataset.save_to_disk(str(save_path))
                self.logger.info(f"✓ Saved {source_info['dataset_name']} to {save_path}")
                return True
        except Exception as e:
            self.logger.warning(f"{source_info['dataset_name']} failed: {e}")
        return False
    
    _OPUS_CORPUS_NAMES = {
        'opensubtitles': 'OpenSubtitles',
        'books': 'Books',
        'multium': 'MultiUN',
        'tatoeba': 'Tatoeba',
    }

    def _download_direct_opus(self, pair: LanguagePair, output_dir: Path) -> bool:
        """Download directly from opus.nlpl.eu as fallback when HF datasets fail."""
        if self.session is None:
            self.logger.warning("HTTP session unavailable, skipping direct OPUS download")
            return False

        corpora = ['OpenSubtitles', 'ParaCrawl', 'WikiMatrix']
        flat_file = output_dir / f"{pair.pair_string}.txt"

        for corpus in corpora:
            url = self._OPUS_DIRECT_URL.format(corpus=corpus, lang_pair=pair.pair_string)
            try:
                self.logger.info(f"Downloading {corpus} from {url}")
                resp = self.session.get(url, timeout=120, stream=True)
                if resp.status_code != 200:
                    self.logger.debug(f"{corpus} not available for {pair.pair_string} (HTTP {resp.status_code})")
                    continue
                total = 0
                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    for name in zf.namelist():
                        if name.endswith('.txt') or name.endswith('.tsv'):
                            with zf.open(name) as f:
                                content = f.read().decode('utf-8')
                            with open(flat_file, 'a', encoding='utf-8') as out:
                                out.write(content)
                                total += content.count('\n')
                if total > 0:
                    self.logger.info(f"✓ Downloaded {total:,} lines from {corpus} for {pair.pair_string}")
                    return True
            except Exception as e:
                self.logger.warning(f"Direct OPUS download failed for {corpus}/{pair.pair_string}: {e}")
                continue
        return False

    def _download_opus_pair(self,
                           pair: LanguagePair,
                           source_info: Dict,
                           output_dir: Path) -> bool:
        """Download OPUS dataset for language pair"""
        suffix = source_info['dataset_name'].split('/')[-1].replace('opus_', '')
        corpus = self._OPUS_CORPUS_NAMES.get(suffix.lower(), suffix.capitalize())
        return self._download_opus_file(
            corpus,
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


def main():
    """CLI entry point for the unified data downloader"""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Unified Data Downloader")
    parser.add_argument("--config", default="config/base.yaml",
                        help="Path to config file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate download size without downloading")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from config.schemas import load_config
    config = load_config(args.config)

    downloader = UnifiedDataDownloader(config)

    if args.dry_run:
        print("=" * 60)
        print("DRY RUN — Estimating download sizes")
        print("=" * 60)
        estimates = downloader.estimate_download_size()
        print(f"  Estimated total: {estimates.get('total_gb', 0):.2f} GB")
        schedule = downloader.get_download_schedule()
        for batch in schedule:
            print(f"\n  Batch: {batch['batch_name']}")
            print(f"    Pairs: {len(batch['pairs'])}")
            for pair in batch['pairs']:
                print(f"      {pair.pair_string} ({pair.priority.value}, {pair.expected_size:,} sentences)")
        print("\nDry run complete. Pass --dry-run to see this, or omit it to download.")
        return

    stats = downloader.download_all()
    print(f"\nDownload complete: {stats}")


if __name__ == "__main__":
    main()