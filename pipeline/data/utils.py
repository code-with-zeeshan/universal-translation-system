# pipeline/data/utils.py
from pathlib import Path
from typing import Dict, List, Optional, Iterator
import yaml
import torch
# Optional dependency: datasets. For smoke/dry-run we can avoid requiring it.
try:
    from datasets import Dataset, IterableDataset  # type: ignore
except Exception:  # pragma: no cover
    Dataset = None  # type: ignore
    IterableDataset = None  # type: ignore
from tqdm import tqdm
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import gc
import itertools
from utils.exceptions import DataError

# Import from common utils
from utils.common_utils import DirectoryManager


class DataProcessor:
    """Shared data processing functionality"""
    
    def __init__(self, config: dict, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.config = config
    
    def process_streaming_dataset(
        self, 
        dataset: IterableDataset, 
        output_path: Path,
        batch_size: int = 1000,
        max_samples: Optional[int] = None,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
    ) -> int:
        """
        Process streaming dataset and save as tab-separated source\\ttarget text file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        samples_processed = 0
        batch_data = []
        
        try:
            for sample in tqdm(dataset, desc=f"Processing {output_path.name}"):
                batch_data.append(sample)
                samples_processed += 1
                
                if len(batch_data) >= batch_size:
                    self._save_batch(batch_data, output_path, samples_processed, source_lang, target_lang)
                    batch_data = []
                    if samples_processed % (batch_size * 10) == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                if max_samples and samples_processed >= max_samples:
                    break
            
            if batch_data:
                self._save_batch(batch_data, output_path, samples_processed, source_lang, target_lang)
                del batch_data
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            self.logger.info(f"✅ Processed {samples_processed:,} samples to {output_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process dataset: {e}")
            raise DataError(f"Failed to process dataset: {e}") from e
        
        return samples_processed
    
    @staticmethod
    def _extract_text(item: dict, source_lang: Optional[str], target_lang: Optional[str]) -> tuple:
        """Extract source/target text from a dataset sample, handling common formats."""
        if 'translation' in item:
            trans = item['translation']
            if isinstance(trans, dict) and source_lang and target_lang:
                return trans.get(source_lang, ''), trans.get(target_lang, '')
        src = item.get('source_text') or item.get('source') or item.get('src') or ''
        tgt = item.get('target_text') or item.get('target') or item.get('tgt') or ''
        return str(src), str(tgt)
    
    def _save_batch(self, batch_data: List[dict], output_path: Path, total_processed: int,
                    source_lang: Optional[str] = None, target_lang: Optional[str] = None) -> None:
        """Write batch as tab-separated source\\ttarget lines to a flat file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = 'w' if total_processed <= len(batch_data) else 'a'
        with open(output_path, mode, encoding='utf-8') as f:
            for item in batch_data:
                src_text, tgt_text = self._extract_text(item, source_lang, target_lang)
                if src_text and tgt_text:
                    f.write(f"{src_text}\t{tgt_text}\n")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_language_pairs_from_config(self) -> List[tuple]:
        """
        Generate language pairs based on configuration
        
        Returns:
            List of (source, target) language pairs
        """
        pairs = []
        distribution = self.config.data.training_distribution
        
        for pair_str in distribution.keys():
            if '-' in pair_str:
                source, target = pair_str.split('-')
                pairs.append((source, target))
        
        return pairs
    
    def validate_language_pair(self, source: str, target: str) -> bool:
        """
        Validate if a language pair is in the configured languages
        
        Args:
            source: Source language code
            target: Target language code
            
        Returns:
            True if both languages are configured
        """
        languages = self.config.data.active_languages
        return source in languages and target in languages


class DatasetLoader:
    """Centralized dataset loading with error handling"""
    
    def __init__(self, logger: Optional[logging.Logger] = None, cache_dir: Optional[str] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.cache_dir = cache_dir
    
    def load_dataset_safely(
        self,
        dataset_name: str,
        config_name: Optional[str] = None,
        split: Optional[str] = None,
        streaming: bool = False,
        **kwargs
    ) -> Optional[Dataset]:
        """
        Safely load a dataset with error handling and retry/fallback.
        
        Tries with trust_remote_code=True first, then False (some
        Parquet-based datasets reject the flag).
        """
        from datasets import load_dataset
        
        for trust_remote in [True, False]:
            try:
                self.logger.info(f"📥 Loading dataset: {dataset_name}")
                load_args = {
                    'path': dataset_name,
                    'streaming': streaming,
                    'trust_remote_code': trust_remote,
                }
                if config_name:
                    load_args['name'] = config_name
                if split:
                    load_args['split'] = split
                if self.cache_dir:
                    load_args['cache_dir'] = self.cache_dir
                load_args.update(kwargs)
                dataset = load_dataset(**load_args)
                self.logger.info(f"✅ Successfully loaded {dataset_name}")
                return dataset
            except Exception as e:
                if trust_remote:
                    self.logger.debug(f"trust_remote_code=True failed for {dataset_name}, trying False: {e}")
                else:
                    self.logger.error(f"❌ Failed to load {dataset_name}: {e}")
                    raise DataError(f"Failed to load {dataset_name}: {e}") from e


# Utility functions for common operations
def get_corpus_size_mb(file_path: Path) -> float:
    """Get size of corpus file in MB"""
    if file_path.exists():
        return file_path.stat().st_size / (1024 * 1024)
    return 0.0


def estimate_sentence_count(file_path: Path, sample_size: int = 1000) -> int:
    """Estimate number of sentences in a file"""
    if not file_path.exists():
        return 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        sample_lines = list(itertools.islice(f, sample_size))
    
    if not sample_lines:
        return 0

    sample_line_count = len(sample_lines)
    sample_size_bytes = sum(len(line.encode('utf-8')) for line in sample_lines)
    total_size_bytes = file_path.stat().st_size
    
    if sample_size_bytes == 0:
        return 0

    estimated_lines = int(total_size_bytes / sample_size_bytes * sample_line_count)
    return estimated_lines


def merge_datasets(dataset_paths: List[Path], output_path: Path) -> None:
    """Merge multiple dataset files into one"""
    logger = logging.getLogger(__name__)
    
    DirectoryManager.create_directory(output_path.parent)
    
    total_lines = 0
    with open(output_path, 'w', encoding='utf-8') as out_file:
        for dataset_path in dataset_paths:
            if dataset_path.exists():
                logger.info(f"📄 Merging {dataset_path.name}")
                with open(dataset_path, 'r', encoding='utf-8') as in_file:
                    for line in in_file:
                        out_file.write(line)
                        total_lines += 1
    
    logger.info(f"✅ Merged {len(dataset_paths)} files into {output_path} ({total_lines:,} lines)")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def safe_file_write(file_path: Path, content: str, mode: str = 'w') -> None:
    """Write to file with retry logic"""
    with open(file_path, mode, encoding='utf-8') as f:
        f.write(content)