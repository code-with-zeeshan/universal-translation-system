# data/practical_data_pipeline.py
"""
Main pipeline orchestrator - Refactored to integrate all modules
"""

import asyncio
import json
from typing import Dict, List, Optional, Iterator
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import os
from opentelemetry import trace
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from config.schemas import RootConfig

tracer = trace.get_tracer(__name__)
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.0.0")


from pathlib import Path
from typing import Dict, List, Optional
import logging

# Import standalone modules
from download_curated_data import CuratedDataDownloader
from download_training_data import MultilingualDataCollector
from smart_data_downloader import SmartDataStrategy
from smart_sampler import SmartDataSampler
from synthetic_augmentation import SyntheticDataAugmenter
from connector.pipeline_connector import PipelineConnector
from connector.vocabulary_connector import VocabularyConnector

# Import shared utilities
from data_utils import DataProcessor, DatasetLoader
from utils.common_utils import DirectoryManager
from utils.exceptions import DataError
from utils.resource_monitor import resource_monitor

logger = logging.getLogger(__name__)

class PracticalDataPipeline:
    """Main orchestrator for the multilingual data pipeline with enhanced error handling"""
    
    def __init__(self, config: RootConfig):
        with tracer.start_as_current_span("PracticalDataPipeline.__init__") as span:
            span.set_attribute("model_version", MODEL_VERSION)

            self.config = config
            self.logger = logging.getLogger(__name__)
            self.data_processor = DataProcessor(self.config, self.logger)
            
            # Initialize strategy
            self.strategy = SmartDataStrategy()
            self.required_pairs = self.strategy.get_required_pairs()
            
            # Initialize components with error handling
            try:
                self.curated_downloader = CuratedDataDownloader()
                self.training_downloader = MultilingualDataCollector(
                    self.config.data.active_languages
                )
                self.sampler = SmartDataSampler()
                self.augmenter = SyntheticDataAugmenter()
            except Exception as e:
                self.logger.error(f"Failed to initialize components: {e}")
                raise
            
            # Create directory structure
            self.dirs = DirectoryManager.create_data_structure(
                self.config.data.processed_dir
            )
            
            # Track pipeline state
            self.pipeline_state = {
                'evaluation_data': False,
                'training_data': False,
                'sampled_data': False,
                'augmented_data': False,
                'training_ready': False,
                'validated': False
            }
            
            self.logger.info("✅ Pipeline initialized with all components")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    def _download_with_retry(self, download_func, *args, **kwargs):
        """Wrapper for download functions with retry logic"""
        return download_func(*args, **kwargs)
    
    async def prepare_all_data(self, resume: bool = True) -> None: 
        """Execute complete data pipeline with integrated modules and resume capability"""
        with tracer.start_as_current_span("PracticalDataPipeline.prepare_all_data") as span:
            span.set_attribute("model_version", MODEL_VERSION)
            span.set_attribute("resume", resume)

            self.logger.info("🚀 Starting integrated data pipeline")

            # Load checkpoint if resuming
            if resume:
                self._load_checkpoint()

            # Execute pipeline steps
            pipeline_steps = [
                ("evaluation_data", "Downloading evaluation data", self._download_evaluation_data_async),
                ("training_data", "Downloading training data", self._download_training_data_async),
                ("sampled_data", "Sampling and filtering data", self._sample_and_filter_data_async),
                ("augmented_data", "Augmenting with synthetic data", self._augment_synthetic_data_async),
                ("training_ready", "Creating training-ready data", self._create_training_ready_data_async),
                ("validated", "Validating final dataset", self._validate_final_dataset_async)
            ]
            
            for state_key, description, step_func in pipeline_steps:
                if not self.pipeline_state.get(state_key, False):
                    self.logger.info(f"\n{'='*60}")
                    self.logger.info(f"📌 {description}")
                    self.logger.info('='*60)
                    
                    try:
                        await step_func()
                        self.pipeline_state[state_key] = True
                        self._save_checkpoint()
                    except Exception as e:
                        self.logger.error(f"❌ Failed: {description} - {e}")
                        span.record_exception(e)
                        raise
                else:
                    self.logger.info(f"⏩ Skipping {description} (already completed)")
            
            self.logger.info("✅ Pipeline completed successfully!")
            span.set_attribute("pipeline.completed", True)

            summary = resource_monitor.get_summary()
            self.logger.info(f"Data pipeline resource summary: {summary}")

    def _save_checkpoint(self):
        """Save pipeline state for resume capability"""
        checkpoint_path = self.dirs['base'] / 'pipeline_checkpoint.json'
        with open(checkpoint_path, 'w') as f:
            json.dump(self.pipeline_state, f)
        self.logger.debug(f"Saved checkpoint: {self.pipeline_state}")
    
    def _load_checkpoint(self):
        """Load pipeline state if exists"""
        checkpoint_path = self.dirs['base'] / 'pipeline_checkpoint.json'
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r') as f:
                self.pipeline_state = json.load(f)
            self.logger.info(f"Loaded checkpoint: {self.pipeline_state}")

    async def _download_evaluation_data_async(self) -> None: 
        """Download evaluation data using CuratedDataDownloader with retry and validation"""
        with resource_monitor.monitor("download_evaluation_data"):
            with tracer.start_as_current_span("download_evaluation_data") as span:
                self.logger.info("📥 Step 1: Downloading evaluation data")
            
                try:
                    # Use retry wrapper
                    stats = self._download_with_retry(
                        self.curated_downloader.download_essential_data,
                        output_dir=str(self.dirs['essential'])
                    )

                    # Validate downloaded data
                    if stats['total_files'] == 0:
                        raise DataError("No evaluation data downloaded")
                
                    span.set_attribute("evaluation_data.files", stats['total_files'])
                    span.set_attribute("evaluation_data.size_mb", stats['total_size_mb'])
                
                except Exception as e:
                    self.logger.error(f"❌ Failed to download evaluation data: {e}")
                    span.record_exception(e)
                    raise

    async def prepare_all_data_async(self) -> None: 
        """Async version of pipeline for better performance"""
        self.logger.info("🚀 Starting async data pipeline")
        
        # Run download tasks concurrently where possible
        download_tasks = [
            self._download_evaluation_data_async(),
            self._download_training_data_async()
        ]
        
        await asyncio.gather(*download_tasks)
        
        # Sequential steps that depend on downloads
        await self._sample_and_filter_data_async()
        await self._augment_synthetic_data_async()
        await self._create_training_ready_data_async()
        await self._validate_final_dataset_async()
        
        self.logger.info("✅ Async pipeline completed!")
    
    async def _download_training_data_async(self) -> None: 
        """Download training data based on strategy"""
        with resource_monitor.monitor("download_training_data"):
            with tracer.start_as_current_span("PracticalDataPipeline._download_training_data") as span:
                span.set_attribute("model_version", MODEL_VERSION)
                self.logger.info("📥 Step 2: Downloading training data based on strategy")
            
                # Group pairs by priority
                high_priority = [p for p in self.required_pairs if p.priority == 'high']
                medium_priority = [p for p in self.required_pairs if p.priority == 'medium']
                low_priority = [p for p in self.required_pairs if p.priority == 'low']
            
                # Download in priority order
                for priority_group, pairs in [
                    ("HIGH", high_priority),
                    ("MEDIUM", medium_priority),
                    ("LOW", low_priority)
                ]:
                    self.logger.info(f"📊 Downloading {priority_group} priority pairs ({len(pairs)} pairs)")
                
                    for pair in pairs:
                        try:
                            self._download_language_pair(pair)
                        except Exception as e:
                            self.logger.error(f"❌ Failed to download {pair.source}-{pair.target}: {e}")
                            continue
    
    def _download_language_pair(self, pair) -> None: 
        """Download a specific language pair"""
        output_file = self.dirs['raw'] / f"{pair.source}-{pair.target}.txt"
        
        # Skip if already exists and is reasonable size
        if output_file.exists():
            size_mb = output_file.stat().st_size / (1024 * 1024)
            if size_mb > 10:  # Skip if > 10MB
                self.logger.info(f"⏩ Skipping {pair.source}-{pair.target} (already exists: {size_mb:.1f}MB)")
                return
        
        # Use training downloader's methods
        self.logger.info(f"📥 Downloading {pair.source}-{pair.target} (priority: {pair.priority})")
        
        # Call the training downloader's specific pair method
        success = self.training_downloader.download_specific_pair(
            source=pair.source,
            target=pair.target,
            output_dir=str(self.dirs['raw'])
        )
    
        if not success:
            self.logger.warning(f"⚠️  Failed to download {pair.source}-{pair.target}")
       
    
    async def _sample_and_filter_data_async(self) -> None: 
        """Sample and filter downloaded data"""
        with resource_monitor.monitor("sample_and_filter_data"):
            with tracer.start_as_current_span("PracticalDataPipeline._sample_and_filter_data") as span:
                span.set_attribute("model_version", MODEL_VERSION)
                self.logger.info("🔍 Step 3: Sampling and filtering data")
            
                distribution = self.config.data.training_distribution
            
                for pair_str, target_size in distribution.items():
                    source, target = pair_str.split('-')
                    input_file = self.dirs['raw'] / f"{pair_str}.txt"
                    output_file = self.dirs['sampled'] / f"{pair_str}_sampled.txt"
                
                    if input_file.exists():
                        self.logger.info(f"📊 Sampling {pair_str}: target {target_size:,} sentences")
                        sampler = SmartDataSampler()
                        stats = sampler.sample_high_quality_pairs(
                            input_file=str(input_file),
                            output_file=str(output_file),
                            target_size=target_size,
                            source_lang=source,  
                            target_lang=target
                        )
                    
                        self.logger.info(f"✅ Sampled {pair_str}: {stats['written_count']:,} sentences")
                    else:
                        self.logger.warning(f"⚠️  Input file not found for {pair_str}")
    
    async def _augment_synthetic_data_async(self) -> None: 
        """Augment with synthetic data"""
        with resource_monitor.monitor("augment_synthetic_data"):
            with tracer.start_as_current_span("PracticalDataPipeline._augment_synthetic_data") as span:
                span.set_attribute("model_version", MODEL_VERSION)
                self.logger.info("🤖 Step 4: Augmenting with synthetic data")
            
                # Augment major language pairs
                major_pairs = ['en-es', 'en-fr', 'en-de', 'en-zh', 'en-ru']
            
                for pair_str in major_pairs:
                    source, target = pair_str.split('-')
                
                    # Check if we have monolingual data
                    mono_file = self.dirs['raw'] / f"mono_{source}.txt"
                    output_file = self.dirs['final'] / f"augmented_{pair_str}.txt"
                
                    if mono_file.exists():
                        self.logger.info(f"🔄 Augmenting {pair_str} with backtranslation")
                    
                        self.augmenter.augment_with_backtranslation(
                            monolingual_file=str(mono_file),
                            source_lang=source,
                            target_lang=target,
                            output_file=str(output_file)
                        )
            
                # Generate pivot translations
                self.logger.info("🔄 Generating pivot translations")
                self.augmenter.generate_pivot_translations(
                    english_pairs_dir=str(self.dirs['sampled'])
                )

    async def _create_training_ready_data_async(self) -> None: 
        """Create data ready for training"""
        with resource_monitor.monitor("create_training_ready_data"):
            with tracer.start_as_current_span("PracticalDataPipeline._create_training_ready_data") as span:
                span.set_attribute("model_version", MODEL_VERSION)
                self.logger.info("📝 Step 5: Creating training-ready data")
        
                #from pipeline_connector import PipelineConnector
                connector = PipelineConnector(self.config)
        
                # Create monolingual corpora for vocabulary
                self.logger.info("Creating monolingual corpora...")
                connector.create_monolingual_corpora()
        
                # Create final training file
                self.logger.info("Creating final training file...")
                connector.create_final_training_file()

    def _create_vocabulary_packs(self) -> None: 
        """Create vocabulary packs from processed data"""
        self.logger.info("📚 Step 6: Creating vocabulary packs")
    
        vocab_connector = VocabularyConnector()
        created_packs = vocab_connector.create_vocabularies_from_pipeline(
            processed_dir=str(self.dirs['processed'])
        )
    
        self.logger.info(f"✅ Created {len(created_packs)} vocabulary packs")       
    
    async def _validate_final_dataset_async(self) -> None: 
        """Validate the final dataset"""
        with resource_monitor.monitor("validate_final_dataset"):
            with tracer.start_as_current_span("PracticalDataPipeline._validate_final_dataset") as span:
                span.set_attribute("model_version", MODEL_VERSION)
                self.logger.info("✔️  Step 5: Validating final dataset")
            
                total_size_gb = 0
                total_sentences = 0
            
                # Check all output directories
                for directory in [self.dirs['sampled'], self.dirs['final']]:
                    for file_path in directory.glob("*.txt"):
                        size_gb = file_path.stat().st_size / (1024**3)
                        total_size_gb += size_gb
                    
                        # Estimate sentences
                        with open(file_path, 'r', encoding='utf-8') as f:
                            sentences = sum(1 for _ in f)
                        total_sentences += sentences
                    
                        self.logger.info(f"📊 {file_path.name}: {sentences:,} sentences ({size_gb:.2f}GB)")
            
                self.logger.info(f"📈 Total dataset: {total_sentences:,} sentences ({total_size_gb:.2f}GB)")
            
                # Validate against target
                target_gb = self.config.data.total_size_gb
                if total_size_gb < target_gb * 0.9:
                    self.logger.warning(f"⚠️  Dataset size ({total_size_gb:.2f}GB) is below target ({target_gb}GB)")
                else:
                    self.logger.info(f"✅ Dataset size meets target!")

    def validate_data_integrity(self) -> Dict[str, bool]:
        """Validate data integrity across pipeline stages"""
        validations = {
            'raw_data_exists': any(self.dirs['raw'].glob('*.txt')),
            'sampled_data_valid': self._validate_sampled_format(),
            'language_coverage': self._check_language_coverage(),
            'size_requirements': self._check_size_requirements()
        }
        return validations 

    def get_pipeline_progress(self) -> Dict[str, float]:
        """Get overall pipeline progress"""
        total_steps = len(self.pipeline_state)
        completed_steps = sum(1 for v in self.pipeline_state.values() if v)
    
        return {
            'overall_progress': (completed_steps / total_steps) * 100,
            'current_stage': self._get_current_stage(),
            'estimated_time_remaining': self._estimate_time_remaining()
        }   


def main():
    """Main entry point"""
    from utils.logging_config import setup_logging
    from config.schemas import load_config

    setup_logging(log_dir="logs/training", log_level="INFO")
    
    # Load configuration using the new system
    config = load_config('config/base.yaml')

    pipeline = PracticalDataPipeline(config)
    asyncio.run(pipeline.prepare_all_data())


if __name__ == "__main__":
    main()
