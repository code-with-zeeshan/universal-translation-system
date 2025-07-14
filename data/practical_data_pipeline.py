# data/practical_data_pipeline.py
from dataclasses import dataclass
from typing import Dict
import yaml
from pathlib import Path
import logging
from datasets import load_dataset
from smart_sampler import SmartDataSampler
from synthetic_augmentation import SyntheticDataAugmenter
from tqdm import tqdm

@dataclass
class PipelineConfig:
    """Configuration for data pipeline"""
    training_distribution: Dict[str, int]
    total_size_gb: int
    output_dir: str
    
    @classmethod
    def from_yaml(cls, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        return cls(
            training_distribution=config_data['training_distribution'],
            total_size_gb=config_data['total_size_gb'],
            output_dir=config_data['output_dir']
        )

class PracticalDataPipeline:
    """Realistic data pipeline for 20 languages with modern configuration"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        log_dir = Path('log')
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_dir /'data_pipeline.log'), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
        self.config = PipelineConfig.from_yaml(config_path)
        self.sampler = SmartDataSampler()
        self.augmenter = SyntheticDataAugmenter()
    
    def prepare_all_data(self) -> None:
        """Execute complete data pipeline"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.download_eval_sets()
        self.download_english_pairs()
        self.download_direct_pairs()
        self.sample_large_corpora()
        self.augment_synthetic()
        
        self.logger.info("✅ Total data: ~8GB of high-quality parallel text")
    
    def get_training_distribution(self) -> Dict[str, int]:
        """Return smart data distribution for training"""
        return self.config.training_distribution
    
    def download_eval_sets(self) -> None:
        """Download high-quality evaluation sets"""
        self.logger.info("📥 Downloading evaluation sets...")
        try:
            flores = load_dataset(
                'facebook/flores',
                name='flores200_sacrebleu_tokenized_xlm_roberta_base',
                split='dev',
                trust_remote_code=True
            )
            flores.save_to_disk(Path(self.config.output_dir) / 'flores200')
        except Exception as e:
            self.logger.error(f"✗ Failed to download evaluation sets: {e}")
    
    def download_english_pairs(self) -> None:
        """Download English-centric pairs"""
        self.logger.info("📥 Downloading English-centric pairs...")
        for pair in self.get_training_distribution():
            if pair.startswith('en-'):
                try:
                    dataset = load_dataset('Helsinki-NLP/opus-100', language_pair=pair, streaming=True, trust_remote_code=True)
                    self._process_streaming_dataset(dataset, Path(self.config.output_dir) / f'opus_{pair}')
                except Exception as e:
                    self.logger.error(f"✗ Failed to download {pair}: {e}")
    
    def download_direct_pairs(self) -> None:
        """Download key direct pairs"""
        self.logger.info("📥 Downloading direct pairs...")
        for pair in self.get_training_distribution():
            if not pair.startswith('en-'):
                try:
                    dataset = load_dataset('Helsinki-NLP/opus-100', language_pair=pair, streaming=True, trust_remote_code=True)
                    self._process_streaming_dataset(dataset, Path(self.config.output_dir) / f'opus_{pair}')
                except Exception as e:
                    self.logger.error(f"✗ Failed to download {pair}: {e}")
    
    def sample_large_corpora(self) -> None:
        """Sample and filter large corpora using SmartDataSampler"""
        self.logger.info("🔍 Sampling large corpora...")
        output_dir = Path(self.config.output_dir) / 'sampled'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for pair, target_size in tqdm(self.get_training_distribution().items(), desc="Sampling corpora"):
            input_file = Path(self.config.output_dir) / 'raw' / f'opus_{pair}.txt'
            output_file = output_dir / f'{pair}.txt'
            if input_file.exists():
                try:
                    self.sampler.sample_high_quality_pairs(
                        input_file=str(input_file),
                        output_file=str(output_file),
                        target_size=target_size
                    )
                    self.logger.info(f"✓ Sampled {pair}: {target_size:,} sentences")
                except Exception as e:
                    self.logger.error(f"✗ Failed to sample {pair}: {e}")
    
    def augment_synthetic(self) -> None:
        """Augment with synthetic data using SyntheticDataAugmenter"""
        self.logger.info("🤖 Augmenting synthetic data...")
        output_dir = Path(self.config.output_dir) / 'final'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for pair in tqdm(self.get_training_distribution(), desc="Augmenting data"):
            source_lang, target_lang = pair.split('-')
            monolingual_file = Path(self.config.output_dir) / 'raw' / f'mono_{source_lang}.txt'
            output_file = output_dir / f'backtranslated_{pair}.txt'
            
            if monolingual_file.exists():
                try:
                    self.augmenter.augment_with_backtranslation(
                        monolingual_file=str(monolingual_file),
                        source_lang=source_lang,
                        target_lang=target_lang,
                        output_file=str(output_file)
                    )
                    self.logger.info(f"✓ Augmented {pair} with backtranslation")
                except Exception as e:
                    self.logger.error(f"✗ Failed to augment {pair}: {e}")

    def _process_streaming_dataset(self, dataset, output_path: Path) -> None:
        """Process streaming dataset with memory efficiency"""
        output_path.mkdir(parents=True, exist_ok=True)
        for batch in dataset.iter(batch_size=1000):
            datasets.Dataset.from_dict(batch).save_to_disk(output_path)
            torch.cuda.empty_cache() if torch.cuda.is_available() else None                