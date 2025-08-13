import torch
from torch.utils.data import Sampler, Dataset
from collections import defaultdict
import numpy as np
from typing import List, Dict, Iterator, Optional, Tuple, Callable
from pathlib import Path
import yaml
import logging
import threading
from tenacity import retry, stop_after_attempt, wait_exponential
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
import zipfile
import io
import mmap
import asyncio
import aiofiles
import itertools
from concurrent.futures import ProcessPoolExecutor
import re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer, util
from dataclasses import dataclass
import random

# --- Configuration Management ---

class ConfigManager:
    """Centralized configuration management with thread-safe singleton pattern."""
    _config = None
    _config_path = None
    _lock = threading.Lock()

    @classmethod
    def load_config(cls, config_path: str = 'data/config.yaml') -> dict:
        """Load configuration from YAML file."""
        with cls._lock:
            if cls._config is None or cls._config_path != config_path:
                with open(config_path, 'r', encoding='utf-8') as f:
                    cls._config = yaml.safe_load(f)
                cls._config_path = config_path
        return cls._config

    @classmethod
    def get_config(cls) -> dict:
        """Get the loaded configuration."""
        if cls._config is None:
            raise ValueError("Configuration not loaded. Call load_config first.")
        return cls._config

# --- Utility Classes ---



class DirectoryManager:
    """Manages directory creation."""
    @staticmethod
    def create_directory(path: Path) -> Path:
        """Creates a directory if it doesn't exist."""
        path.mkdir(parents=True, exist_ok=True)
        return path

# --- Custom Samplers ---

class TemperatureSampler(Sampler[List[int]]):
    """
    Samples elements from multiple language groups with temperature scaling.
    This sampler is designed to work with a `DataLoader` and yields batches of indices.
    """
    def __init__(self, dataset: Dataset, batch_size: int, temperature: float = 1.0, drop_last: bool = False):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.temperature = temperature
        self.drop_last = drop_last

        if not hasattr(dataset, 'get_lang_pair_indices'):
            raise ValueError("Dataset must have a 'get_lang_pair_indices' method.")
        
        self.lang_pair_indices = self.dataset.get_lang_pair_indices()
        self.lang_pairs = list(self.lang_pair_indices.keys())
        
        base_probs = torch.tensor([len(indices) for indices in self.lang_pair_indices.values()], dtype=torch.float)
        
        if self.temperature != 1.0:
            scaled_logits = torch.log(base_probs) / self.temperature
            self.sampling_weights = torch.softmax(scaled_logits, dim=0)
        else:
            self.sampling_weights = base_probs / base_probs.sum()

        self.num_samples = len(self.dataset)
        if self.drop_last and self.num_samples % self.batch_size != 0:
            self.num_batches = self.num_samples // self.batch_size
        else:
            self.num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.num_batches):
            batch_indices = []
            for _ in range(self.batch_size):
                chosen_pair = self.lang_pairs[torch.multinomial(self.sampling_weights, 1).item()]
                chosen_index = random.choice(self.lang_pair_indices[chosen_pair])
                batch_indices.append(chosen_index)
            yield batch_indices

    def __len__(self) -> int:
        return self.num_batches

# --- Data Processing ---

class DataProcessor:
    """Shared data processing functionality."""
    def __init__(self, config: dict, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.config = config

    def process_streaming_dataset(
        self,
        dataset: torch.utils.data.IterableDataset,
        output_path: Path,
        batch_size: int = 1000,
        max_samples: Optional[int] = None
    ) -> int:
        """Process streaming dataset with memory efficiency and proper cleanup."""
        DirectoryManager.create_directory(output_path.parent)
        samples_processed = 0
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for sample in tqdm(dataset, desc=f"Processing {output_path.name}"):
                # Assuming the dataset yields dictionaries with 'source' and 'target' keys
                f_out.write(f"{sample['source']}\t{sample['target']}\n")
                samples_processed += 1
                if max_samples and samples_processed >= max_samples:
                    break
        self.logger.info(f"✅ Processed {samples_processed:,} samples to {output_path}")
        return samples_processed

# --- Data Downloading ---

class DataDownloader:
    """Downloads data from various sources."""
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Creates a requests session with retry logic."""
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

    def download_file(self, url: str, output_path: Path, description: Optional[str] = None):
        """Downloads a file with progress bar."""
        with self.session.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(output_path, 'wb') as f, tqdm(
                desc=description or url.split('/')[-1],
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)

# --- Main Data Manager ---

class DataManager:
    """Main class for managing the data pipeline."""
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_downloader = DataDownloader(self.logger)
        self.data_processor = DataProcessor(self.config, self.logger)

    def run_pipeline(self):
        """Runs the entire data pipeline."""
        self.logger.info("Starting data pipeline...")
        # 1. Download data
        # 2. Process data
        # 3. Augment data
        # 4. Sample data
        self.logger.info("Data pipeline finished.")

if __name__ == '__main__':
    # This part needs to be updated to load the new config
    # For now, it's left as is, but it will not run without a config object.
    # Example of how it might be run:
    # from config.schemas import load_config
    # config = load_config('config/base.yaml')
    # manager = DataManager(config)
    # manager.run_pipeline()
    pass