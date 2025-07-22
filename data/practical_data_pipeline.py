# data/practical_data_pipeline.py
"""
Main pipeline orchestrator - Refactored to integrate all modules
"""

import os
from opentelemetry import trace
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
tracer = trace.get_tracer(__name__)
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.0.0")
CONFIG_PATH = os.environ.get("PIPELINE_CONFIG_PATH", "data/config.yaml")

class ConfigReloadHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith(CONFIG_PATH):
            with tracer.start_as_current_span("pipeline_config_reload"):
                print("[Pipeline] Config file changed, reloading...")
                # Reload config logic here

observer = Observer()
observer.schedule(ConfigReloadHandler(), path=os.path.dirname(CONFIG_PATH) or '.', recursive=False)
observer.start()

from pathlib import Path
from typing import Dict, List, Optional
import logging

# Import standalone modules
from download_curated_data import CuratedDataDownloader
from download_training_data import MultilingualDataCollector
from smart_data_downloader import SmartDataStrategy
from smart_sampler import SmartDataSampler
from synthetic_augmentation import SyntheticDataAugmenter
from pipeline_connector import PipelineConnector
from data.vocabulary_connector import VocabularyConnector

# Import shared utilities
from data_utils import ConfigManager, DataProcessor, DatasetLoader
from utils.common_utils import StandardLogger, DirectoryManager


class PracticalDataPipeline:
    """Main orchestrator for the multilingual data pipeline"""
    
    def __init__(self, config_path: str = CONFIG_PATH):
        with tracer.start_as_current_span("PracticalDataPipeline.__init__") as span:
            span.set_attribute("model_version", MODEL_VERSION)
            # Initialize logging
            self.logger = StandardLogger.get_logger(__name__)
            StandardLogger.log_system_info(self.logger)
            
            # Load configuration
            self.config = ConfigManager.load_config(config_path)
            self.data_processor = DataProcessor(self.logger)
            
            # Initialize strategy
            self.strategy = SmartDataStrategy()
            self.required_pairs = self.strategy.get_required_pairs()
            
            # Initialize components
            self.curated_downloader = CuratedDataDownloader()
            self.training_downloader = MultilingualDataCollector(
                ConfigManager.get_languages()
            )
            self.sampler = SmartDataSampler()
            self.augmenter = SyntheticDataAugmenter()
            
            # Create directory structure
            self.dirs = DirectoryManager.create_data_structure(
                ConfigManager.get_output_dir()
            )
            
            self.logger.info("✅ Pipeline initialized with all components")
    
    def prepare_all_data(self) -> None:
        """Execute complete data pipeline with integrated modules"""
        with tracer.start_as_current_span("PracticalDataPipeline.prepare_all_data") as span:
            span.set_attribute("model_version", MODEL_VERSION)
            self.logger.info("🚀 Starting integrated data pipeline")
            
            # Step 1: Download evaluation data using curated downloader
            self._download_evaluation_data()
            
            # Step 2: Download training data based on strategy
            self._download_training_data()
            
            # Step 3: Sample and filter data
            self._sample_and_filter_data()
            
            # Step 4: Augment with synthetic data
            self._augment_synthetic_data()

            # Step 5: Create training-ready data
            self._create_training_ready_data()
            
            # Step 6: Validate final dataset
            self._validate_final_dataset()
            
            self.logger.info("✅ Pipeline completed successfully!")
    
    def _download_evaluation_data(self) -> None:
        """Download evaluation data using CuratedDataDownloader"""
        with tracer.start_as_current_span("PracticalDataPipeline._download_evaluation_data") as span:
            span.set_attribute("model_version", MODEL_VERSION)
            self.logger.info("📥 Step 1: Downloading evaluation data")
            
            try:
                # Use the curated downloader
                self.curated_downloader.download_essential_data(
                    output_dir=str(self.dirs['essential'])
                )
            except Exception as e:
                self.logger.error(f"❌ Failed to download evaluation data: {e}")
                raise
    
    def _download_training_data(self) -> None:
        """Download training data based on strategy"""
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
       
    
    def _sample_and_filter_data(self) -> None:
        """Sample and filter downloaded data"""
        with tracer.start_as_current_span("PracticalDataPipeline._sample_and_filter_data") as span:
            span.set_attribute("model_version", MODEL_VERSION)
            self.logger.info("🔍 Step 3: Sampling and filtering data")
            
            distribution = ConfigManager.get_training_distribution()
            
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
    
    def _augment_synthetic_data(self) -> None:
        """Augment with synthetic data"""
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

    def _create_training_ready_data(self) -> None:
        """Create data ready for training"""
        with tracer.start_as_current_span("PracticalDataPipeline._create_training_ready_data") as span:
            span.set_attribute("model_version", MODEL_VERSION)
            self.logger.info("📝 Step 5: Creating training-ready data")
        
            #from pipeline_connector import PipelineConnector
            connector = PipelineConnector()
        
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
    
    def _validate_final_dataset(self) -> None:
        """Validate the final dataset"""
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
            target_gb = self.config.get('total_size_gb', 8)
            if total_size_gb < target_gb * 0.9:
                self.logger.warning(f"⚠️  Dataset size ({total_size_gb:.2f}GB) is below target ({target_gb}GB)")
            else:
                self.logger.info(f"✅ Dataset size meets target!")


def main():
    """Main entry point"""
    pipeline = PracticalDataPipeline()
    pipeline.prepare_all_data()


if __name__ == "__main__":
    main()