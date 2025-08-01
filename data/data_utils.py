# data/data_utils.py
"""
Shared data processing utilities for the multilingual pipeline
Centralizes common data operations to eliminate duplication
"""

from pathlib import Path
from typing import Dict, List, Optional, Iterator
import yaml
import torch
from datasets import Dataset, IterableDataset
from tqdm import tqdm
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

# Import from common utils
from utils.common_utils import StandardLogger, DirectoryManager


class ConfigManager:
    """Centralized configuration management"""
    
    _config = None
    _config_path = None
    
    @classmethod
    def load_config(cls, config_path: str = 'data/config.yaml') -> dict:
        """
        Load configuration from YAML file (singleton pattern)
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        if cls._config is None or cls._config_path != config_path:
            with open(config_path, 'r', encoding='utf-8') as f:
                cls._config = yaml.safe_load(f)
            cls._config_path = config_path
        return cls._config
    
    @classmethod
    def get_languages(cls) -> List[str]:
        """Get list of languages from config"""
        config = cls.load_config()
        return config.get('languages', [])
    
    @classmethod
    def get_training_distribution(cls) -> Dict[str, int]:
        """Get training distribution from config"""
        config = cls.load_config()
        return config.get('training_distribution', {})
    
    @classmethod
    def get_quality_threshold(cls) -> float:
        """Get quality threshold from config"""
        config = cls.load_config()
        return config.get('quality_threshold', 0.8)
    
    @classmethod
    def get_output_dir(cls) -> str:
        """Get output directory from config"""
        config = cls.load_config()
        return config.get('output_dir', 'data/processed')

    @classmethod
    def validate_config(cls) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        config = cls.load_config()
    
        # Check required fields
        required_fields = ['languages', 'max_sentence_length', 'quality_threshold', 
                          'output_dir', 'training_distribution']
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
    
        # Validate language codes
        if 'languages' in config:
            valid_langs = set(config['languages'])
        
            # Check training distribution
            if 'training_distribution' in config:
               for pair_str in config['training_distribution']:
                   if '-' in pair_str:
                        src, tgt = pair_str.split('-')
                        if src not in valid_langs:
                           errors.append(f"Unknown source language in pair {pair_str}: {src}")
                        if tgt not in valid_langs:
                            errors.append(f"Unknown target language in pair {pair_str}: {tgt}")
    
        # Validate numeric values
        if 'quality_threshold' in config:
            if not 0 <= config['quality_threshold'] <= 1:
                errors.append(f"quality_threshold must be between 0 and 1")
    
        return errors    


class DataProcessor:
    """Shared data processing functionality"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or StandardLogger.get_logger(__name__)
        self.config = ConfigManager.load_config()
    
    def process_streaming_dataset(
        self, 
        dataset: IterableDataset, 
        output_path: Path,
        batch_size: int = 1000,
        max_samples: Optional[int] = None
    ) -> int:
        """
        Process streaming dataset with memory efficiency and proper cleanup

        This function processes large datasets in batches to avoid memory issues,
        with automatic garbage collection and GPU cache clearing.        
        
        Args:
            dataset: Streaming dataset from HuggingFace
            output_path: Path to save processed data
            batch_size: Number of samples to process at once (default: 1000)
            max_samples: Maximum number of samples to process (default: None - process all)
            
        Returns:
            int: Number of samples processed

        Raises:
            DataError: If dataset processing fails
            IOError: If output path is not writable    
        """
        output_path = Path(output_path)
        DirectoryManager.create_directory(output_path.parent)
        
        samples_processed = 0
        batch_data = []
        
        try:
            # Process dataset in batches
            for sample in tqdm(dataset, desc=f"Processing {output_path.name}"):
                batch_data.append(sample)
                samples_processed += 1
                
                # Save batch when full
                if len(batch_data) >= batch_size:
                    self._save_batch(batch_data, output_path, samples_processed)

                    # Proper cleanup - delete variables before empty_cache
                    batch_data = []  # Clear the list

                    # Force garbage collection periodically
                    if samples_processed % (batch_size * 10) == 0:
                        import gc
                        gc.collect()
                    
                        # Clear GPU cache if available
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache() 
                
                # Stop if max_samples reached
                if max_samples and samples_processed >= max_samples:
                    break
            
            # Save remaining data
            if batch_data:
                self._save_batch(batch_data, output_path, samples_processed)
                # Final cleanup
                del batch_data

                import gc
                gc.collect()
            
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            self.logger.info(f"✅ Processed {samples_processed:,} samples to {output_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process dataset: {e}")
            raise
        
        return samples_processed
    
    def _save_batch(self, batch_data: List[dict], output_path: Path, total_processed: int) -> None:
        """Save a batch of data to disk with proper memory management"""
        # Create dataset from batch
        batch_dataset = Dataset.from_list(batch_data)
        
        # Save with appropriate naming
        batch_path = output_path / f"batch_{total_processed}"
        batch_dataset.save_to_disk(str(batch_path))

        # Proper memory cleanup - delete before empty_cache
        del batch_dataset
        #del batch_data

        # Force garbage collection
        import gc
        gc.collect()
    
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_language_pairs_from_config(self) -> List[tuple]:
        """
        Generate language pairs based on configuration
        
        Returns:
            List of (source, target) language pairs
        """
        pairs = []
        distribution = ConfigManager.get_training_distribution()
        
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
        languages = ConfigManager.get_languages()
        return source in languages and target in languages


class DatasetLoader:
    """Centralized dataset loading with error handling"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or StandardLogger.get_logger(__name__)
    
    def load_dataset_safely(
        self,
        dataset_name: str,
        config_name: Optional[str] = None,
        split: Optional[str] = None,
        streaming: bool = False,
        **kwargs
    ) -> Optional[Dataset]:
        """
        Safely load a dataset with error handling
        
        Args:
            dataset_name: Name of the dataset
            config_name: Configuration name
            split: Dataset split
            streaming: Whether to use streaming
            **kwargs: Additional arguments for load_dataset
            
        Returns:
            Dataset object or None if failed
        """
        try:
            from datasets import load_dataset
            
            self.logger.info(f"📥 Loading dataset: {dataset_name}")
            
            # Build arguments
            load_args = {
                'path': dataset_name,
                'streaming': streaming,
                'trust_remote_code': True
            }
            
            if config_name:
                load_args['name'] = config_name
            if split:
                load_args['split'] = split
                
            # Add any additional arguments
            load_args.update(kwargs)
            
            # Load dataset
            dataset = load_dataset(**load_args)
            
            self.logger.info(f"✅ Successfully loaded {dataset_name}")
            return dataset
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load {dataset_name}: {e}")
            return None


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
    
    # Count lines in sample
    with open(file_path, 'r', encoding='utf-8') as f:
        sample_lines = sum(1 for i, _ in enumerate(f) if i < sample_size)
    
    # Estimate total based on file size
    if sample_lines > 0:
        sample_size_bytes = sum(len(line.encode('utf-8')) for i, line in enumerate(open(file_path, 'r', encoding='utf-8')) if i < sample_size)
        total_size_bytes = file_path.stat().st_size
        estimated_lines = int(total_size_bytes / sample_size_bytes * sample_lines)
        return estimated_lines
    
    return 0


def merge_datasets(dataset_paths: List[Path], output_path: Path) -> None:
    """Merge multiple dataset files into one"""
    logger = StandardLogger.get_logger(__name__)
    
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