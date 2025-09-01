# data/unified_data_pipeline.py
"""
Unified data pipeline manager combining all data management functionality.
Replaces: data_manager.py and practical_data_pipeline.py
"""

import asyncio
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Any
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

import torch
from torch.utils.data import Sampler, Dataset
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from opentelemetry import trace
# Optional FS monitoring; not required in dry-run
try:
    from watchdog.observers import Observer  # type: ignore
    from watchdog.events import FileSystemEventHandler  # type: ignore
except Exception:
    Observer = None  # type: ignore
    FileSystemEventHandler = None  # type: ignore

from config.schemas import RootConfig
from utils.exceptions import DataError
from utils.resource_monitor import resource_monitor
from utils.logging_config import setup_logging
from utils.common_utils import DirectoryManager

# Import data modules (these would use the unified versions we created earlier)
from data.unified_data_downloader import UnifiedDataDownloader, DatasetType
from data.smart_sampler import SmartDataSampler
from data.synthetic_augmentation import SyntheticDataAugmenter
from connector.pipeline_connector import PipelineConnector
from connector.vocabulary_connector import VocabularyConnector

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class PipelineStage(Enum):
    """Pipeline execution stages"""
    DOWNLOAD_EVAL = "download_evaluation"
    DOWNLOAD_TRAIN = "download_training"
    SAMPLE_FILTER = "sample_filter"
    AUGMENT = "augment"
    CREATE_READY = "create_ready"
    VALIDATE = "validate"
    VOCABULARY = "vocabulary"


@dataclass
class PipelineState:
    """Track pipeline execution state"""
    completed_stages: Dict[str, bool]
    current_stage: Optional[PipelineStage]
    total_sentences: int = 0
    total_size_gb: float = 0.0
    error_count: int = 0
    
    def is_complete(self) -> bool:
        """Check if all stages are complete"""
        return all(self.completed_stages.values())
    
    def get_progress(self) -> float:
        """Get overall progress percentage"""
        if not self.completed_stages:
            return 0.0
        completed = sum(1 for v in self.completed_stages.values() if v)
        return (completed / len(self.completed_stages)) * 100


# ============= CUSTOM SAMPLERS (from data_manager.py) =============

class TemperatureSampler(Sampler[List[int]]):
    """
    Temperature-based sampling for balanced language pair distribution.
    This is the useful component from data_manager.py.
    """
    
    def __init__(self, 
                 dataset: Dataset, 
                 batch_size: int, 
                 temperature: float = 1.0, 
                 drop_last: bool = False):
        """
        Initialize temperature sampler.
        
        Args:
            dataset: Dataset with language pair indices
            batch_size: Batch size for sampling
            temperature: Temperature for probability scaling (1.0 = proportional)
            drop_last: Whether to drop incomplete last batch
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.temperature = temperature
        self.drop_last = drop_last
        
        # Validate dataset has required method
        if not hasattr(dataset, 'get_lang_pair_indices'):
            raise ValueError("Dataset must have 'get_lang_pair_indices' method")
        
        # Get language pair distribution
        self.lang_pair_indices = dataset.get_lang_pair_indices()
        self.lang_pairs = list(self.lang_pair_indices.keys())
        
        # Calculate sampling weights with temperature
        base_probs = torch.tensor(
            [len(indices) for indices in self.lang_pair_indices.values()], 
            dtype=torch.float
        )
        
        if self.temperature != 1.0:
            # Apply temperature scaling
            scaled_logits = torch.log(base_probs) / self.temperature
            self.sampling_weights = torch.softmax(scaled_logits, dim=0)
        else:
            # Proportional sampling
            self.sampling_weights = base_probs / base_probs.sum()
        
        # Calculate number of batches
        self.num_samples = len(self.dataset)
        if self.drop_last and self.num_samples % self.batch_size != 0:
            self.num_batches = self.num_samples // self.batch_size
        else:
            self.num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches with temperature-based sampling"""
        for _ in range(self.num_batches):
            batch_indices = []
            
            for _ in range(self.batch_size):
                # Sample language pair based on temperature-adjusted weights
                chosen_pair = self.lang_pairs[
                    torch.multinomial(self.sampling_weights, 1).item()
                ]
                
                # Random sample from chosen language pair
                chosen_index = random.choice(self.lang_pair_indices[chosen_pair])
                batch_indices.append(chosen_index)
            
            yield batch_indices
    
    def __len__(self) -> int:
        """Return number of batches"""
        return self.num_batches


class BalancedLanguageSampler(Sampler):
    """
    Additional sampler for strict balanced sampling across language pairs.
    Ensures each language pair gets equal representation.
    """
    
    def __init__(self, dataset: Dataset, batch_size: int):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        
        if not hasattr(dataset, 'get_lang_pair_indices'):
            raise ValueError("Dataset must have 'get_lang_pair_indices' method")
        
        self.lang_pair_indices = dataset.get_lang_pair_indices()
        self.lang_pairs = list(self.lang_pair_indices.keys())
        
        # Create balanced sampling order
        self._create_balanced_order()
    
    def _create_balanced_order(self):
        """Create balanced sampling order"""
        # Interleave samples from each language pair
        iterators = {
            pair: iter(random.sample(indices, len(indices)))
            for pair, indices in self.lang_pair_indices.items()
        }
        
        self.sampling_order = []
        exhausted = set()
        
        while len(exhausted) < len(self.lang_pairs):
            for pair in self.lang_pairs:
                if pair not in exhausted:
                    try:
                        idx = next(iterators[pair])
                        self.sampling_order.append(idx)
                    except StopIteration:
                        exhausted.add(pair)
    
    def __iter__(self):
        """Generate samples in balanced order"""
        for i in range(0, len(self.sampling_order), self.batch_size):
            batch = self.sampling_order[i:i + self.batch_size]
            if batch:
                yield batch
    
    def __len__(self):
        """Return number of batches"""
        return (len(self.sampling_order) + self.batch_size - 1) // self.batch_size


# ============= MAIN PIPELINE ORCHESTRATOR =============

class UnifiedDataPipeline:
    """
    Unified data pipeline orchestrator combining all data management functionality.
    This replaces both data_manager.py and practical_data_pipeline.py.
    """
    
    def __init__(self, config: RootConfig, dry_run: bool = False):
        """
        Initialize unified data pipeline.
        
        Args:
            config: Root configuration object
            dry_run: If True, synthesize tiny local sample files and skip actual downloads
        """
        with tracer.start_as_current_span("UnifiedDataPipeline.__init__") as span:
            self.config = config
            self.logger = logging.getLogger(__name__)
            self.dry_run = dry_run
            
            # Initialize components
            self._initialize_components()
            
            # Create directory structure
            # Use the base data directory (parent of processed_dir) to avoid nested 'processed/processed'
            base_dir = str(Path(self.config.data.processed_dir).parent)
            self.dirs = DirectoryManager.create_data_structure(base_dir)
            
            # If dry_run, synthesize tiny sample data so later stages can run locally
            if self.dry_run:
                try:
                    self._synthesize_tiny_samples()
                    # Also synthesize minimal sampled files expected by PipelineConnector
                    self._synthesize_minimal_sampled()
                except Exception as e:
                    self.logger.warning(f"Dry-run synthesis failed: {e}")
            
            # Initialize pipeline state
            self.state = PipelineState(
                completed_stages={stage.value: False for stage in PipelineStage},
                current_stage=None
            )
            
            # Load checkpoint if exists
            self._load_checkpoint()
            
            span.set_attribute("pipeline.initialized", True)
            self.logger.info("✅ Unified data pipeline initialized")
    
    def _initialize_components(self):
        """Initialize all pipeline components with error handling"""
        try:
            # Use unified components
            self.downloader = UnifiedDataDownloader(self.config)
            self.sampler = SmartDataSampler()
            # Only instantiate augmenter if heavy deps are available; otherwise, create a stub that skips
            try:
                self.augmenter = SyntheticDataAugmenter(self.config)
            except Exception as e:
                self.logger.warning(f"Synthetic augmenter unavailable ({e}); augmentation will be skipped in dry-run")
                self.augmenter = None  # type: ignore
            # Connectors may have optional deps; guard gracefully
            try:
                self.pipeline_connector = PipelineConnector(self.config) if PipelineConnector else None
            except Exception as e:
                self.logger.warning(f"PipelineConnector unavailable ({e}); using limited pipeline features")
                self.pipeline_connector = None
            self.vocab_connector = VocabularyConnector()
            
            # Get strategy from downloader
            self.required_pairs = self.downloader.get_required_pairs()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise DataError(f"Component initialization failed: {e}")
    
    # ============= CHECKPOINT MANAGEMENT =============
    
    def _save_checkpoint(self):
        """Save pipeline state for resume capability"""
        checkpoint_path = self.dirs['base'] / 'pipeline_checkpoint.json'
        
        checkpoint_data = {
            'completed_stages': self.state.completed_stages,
            'current_stage': self.state.current_stage.value if self.state.current_stage else None,
            'total_sentences': self.state.total_sentences,
            'total_size_gb': self.state.total_size_gb,
            'error_count': self.state.error_count
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        self.logger.debug(f"Saved checkpoint: {self.state.get_progress():.1f}% complete")
    
    def _load_checkpoint(self):
        """Load pipeline state if checkpoint exists"""
        checkpoint_path = self.dirs['base'] / 'pipeline_checkpoint.json'
        
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r') as f:
                    checkpoint_data = json.load(f)
                
                self.state.completed_stages = checkpoint_data['completed_stages']
                self.state.total_sentences = checkpoint_data.get('total_sentences', 0)
                self.state.total_size_gb = checkpoint_data.get('total_size_gb', 0.0)
                self.state.error_count = checkpoint_data.get('error_count', 0)
                
                if checkpoint_data.get('current_stage'):
                    self.state.current_stage = PipelineStage(checkpoint_data['current_stage'])
                
                self.logger.info(f"Loaded checkpoint: {self.state.get_progress():.1f}% complete")
                
            except Exception as e:
                self.logger.warning(f"Could not load checkpoint: {e}")
    
    # ============= MAIN PIPELINE EXECUTION =============
    
    async def run_pipeline(self, 
                          resume: bool = True,
                          stages: Optional[List[PipelineStage]] = None) -> Dict[str, Any]:
        """
        Execute the complete data pipeline.
        
        Args:
            resume: Whether to resume from checkpoint
            stages: Specific stages to run (None = all)
            
        Returns:
            Pipeline execution summary
        """
        with tracer.start_as_current_span("UnifiedDataPipeline.run_pipeline") as span:
            span.set_attribute("resume", resume)
            span.set_attribute("dry_run", getattr(self, 'dry_run', False))
            
            self.logger.info("🚀 Starting unified data pipeline")
            start_time = asyncio.get_event_loop().time()
            
            # Determine stages to run
            stages_to_run = stages or list(PipelineStage)
            
            # Execute pipeline stages
            for stage in stages_to_run:
                if not resume or not self.state.completed_stages.get(stage.value, False):
                    await self._execute_stage(stage)
                else:
                    self.logger.info(f"⏩ Skipping {stage.value} (already completed)")
            
            # Calculate execution time
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Generate summary
            summary = self._generate_summary(execution_time)
            
            span.set_attribute("pipeline.completed", True)
            span.set_attribute("pipeline.execution_time", execution_time)
            
            self.logger.info("✅ Pipeline completed successfully!")
            self._log_summary(summary)
            
            return summary

    def _synthesize_tiny_samples(self) -> None:
        """Create minimal local sample files for dry-run mode."""
        base = self.dirs['base']
        raw_dir = self.dirs['raw']
        processed_dir = self.dirs['processed']
        sampled_dir = self.dirs['sampled']
        final_dir = self.dirs['final']
        ready_dir = self.dirs.get('ready', processed_dir / 'ready')
        ready_dir.mkdir(parents=True, exist_ok=True)

        # Ensure directories
        for d in (raw_dir, processed_dir, sampled_dir, ready_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Choose a couple of language pairs
        pairs = ['en-es', 'en-fr']
        tiny_lines = [
            ('hello world', 'hola mundo'),
            ('this is a test', 'esto es una prueba'),
            ('how are you', 'como estas')
        ]

        # Create tiny raw bilingual files (tab-separated: src<TAB>tgt)
        for pair in pairs:
            raw_file = raw_dir / f"{pair}.txt"
            if not raw_file.exists():
                with open(raw_file, 'w', encoding='utf-8') as f:
                    for src, tgt in tiny_lines:
                        f.write(f"{src}\t{tgt}\n")

        # Create minimal single-language corpora for vocab creator
        for lang, lines in [('en', ['hello world', 'this is a test']),
                            ('es', ['hola mundo', 'esto es una prueba'])]:
            corp = processed_dir / f"{lang}_corpus.txt"
            if not corp.exists():
                with open(corp, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))

        # Ensure sampled dir exists, even if empty
        sampled_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("🧪 Dry-run: tiny sample data synthesized")

    def _synthesize_minimal_sampled(self) -> None:
        """Create minimal sampled files so PipelineConnector can proceed in dry-run."""
        sampled_dir = self.dirs['sampled']
        sampled_dir.mkdir(parents=True, exist_ok=True)
        # Create a minimal *_sampled.txt with expected 4 columns: src, tgt, src_lang, tgt_lang
        pairs = ['en-es', 'en-fr']
        tiny_lines = [
            ('hello world', 'hola mundo'),
            ('this is a test', 'esto es una prueba'),
            ('how are you', 'como estas')
        ]
        for pair in pairs:
            src, tgt = pair.split('-')
            file_path = sampled_dir / f"{pair}_sampled.txt"
            if not file_path.exists():
                with open(file_path, 'w', encoding='utf-8') as f:
                    for s, t in tiny_lines:
                        f.write(f"{s}\t{t}\t{src}\t{tgt}\n")
        self.logger.info("🧪 Dry-run: minimal sampled files synthesized")
    
    async def _execute_stage(self, stage: PipelineStage):
        """Execute a single pipeline stage with error handling"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"📌 Starting: {stage.value}")
        self.logger.info('='*60)
        
        self.state.current_stage = stage
        
        try:
            with resource_monitor.monitor(stage.value):
                # Route to appropriate stage handler
                stage_handlers = {
                    PipelineStage.DOWNLOAD_EVAL: self._download_evaluation_data,
                    PipelineStage.DOWNLOAD_TRAIN: self._download_training_data,
                    PipelineStage.SAMPLE_FILTER: self._sample_and_filter_data,
                    PipelineStage.AUGMENT: self._augment_data,
                    PipelineStage.CREATE_READY: self._create_training_ready,
                    PipelineStage.VALIDATE: self._validate_dataset,
                    PipelineStage.VOCABULARY: self._create_vocabulary
                }
                
                # In dry-run mode, skip stages that require network and heavy I/O or heavy optional deps
                if getattr(self, 'dry_run', False) and stage in {PipelineStage.DOWNLOAD_EVAL, PipelineStage.DOWNLOAD_TRAIN, PipelineStage.VOCABULARY}:
                    self.logger.info(f"🧪 Dry-run: Skipping stage {stage.value}")
                    self.state.completed_stages[stage.value] = True
                    self._save_checkpoint()
                    return
                
                handler = stage_handlers.get(stage)
                if handler:
                    await handler()
                    self.state.completed_stages[stage.value] = True
                    self._save_checkpoint()
                else:
                    raise ValueError(f"Unknown stage: {stage}")
                    
        except Exception as e:
            self.state.error_count += 1
            self.logger.error(f"❌ Failed: {stage.value} - {e}")
            self._save_checkpoint()
            raise
    
    # ============= STAGE IMPLEMENTATIONS =============
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    async def _download_evaluation_data(self):
        """Download evaluation datasets or synthesize tiny samples in dry-run"""
        with tracer.start_as_current_span("download_evaluation_data") as span:
            if getattr(self, 'dry_run', False):
                self.logger.info("🧪 Dry-run: Skipping evaluation downloads and using synthesized samples")
                span.set_attribute("dry_run", True)
                return
            
            self.logger.info("📥 Downloading evaluation data...")
            stats = self.downloader.download_all(
                output_dir=str(self.dirs['base']),
                dataset_types=[DatasetType.EVALUATION]
            )
            span.set_attribute("eval_data.files", stats.get('total_files', 0))
            self.logger.info(f"✅ Downloaded {stats.get('total_files', 0)} evaluation datasets")
    
    async def _download_training_data(self):
        """Download training data based on strategy or synthesize in dry-run"""
        with tracer.start_as_current_span("download_training_data") as span:
            if getattr(self, 'dry_run', False):
                self.logger.info("🧪 Dry-run: Skipping training downloads and using synthesized samples")
                span.set_attribute("dry_run", True)
                return
            
            self.logger.info("📥 Downloading training data...")
            
            # Use strategy-based download
            schedule = self.downloader.get_download_schedule(DatasetType.TRAINING)
            for batch in schedule:
                self.logger.info(f"Processing batch: {batch['batch_name']}")
            
            stats = self.downloader.download_all(
                output_dir=str(self.dirs['base']),
                dataset_types=[DatasetType.TRAINING]
            )
            span.set_attribute("training_data.pairs", stats.get('downloaded_pairs', 0))
    
    async def _sample_and_filter_data(self):
        """Sample and filter downloaded data"""
        with tracer.start_as_current_span("sample_and_filter") as span:
            self.logger.info("🔍 Sampling and filtering data...")
            
            distribution = self.config.data.training_distribution
            total_sampled = 0
            
            for pair_str, target_size in distribution.items():
                source, target = pair_str.split('-')
                input_file = self.dirs['raw'] / f"{pair_str}.txt"
                output_file = self.dirs['sampled'] / f"{pair_str}_sampled.txt"
                
                if input_file.exists():
                    stats = self.sampler.sample_high_quality_pairs(
                        input_file=str(input_file),
                        output_file=str(output_file),
                        target_size=target_size,
                        source_lang=source,
                        target_lang=target
                    )
                    total_sampled += stats['written_count']
                    self.logger.info(f"✅ Sampled {pair_str}: {stats['written_count']:,} sentences")
            
            self.state.total_sentences += total_sampled
            span.set_attribute("sampled.total", total_sampled)
    
    async def _augment_data(self):
        """Augment with synthetic data"""
        with tracer.start_as_current_span("augment_data") as span:
            self.logger.info("🤖 Augmenting with synthetic data...")
            
            # Augment major language pairs
            augmented_total = 0

            # If augmenter is unavailable or we're in dry-run without heavy deps, skip
            if self.augmenter is None or getattr(self, 'dry_run', False):
                self.logger.info("🧪 Dry-run or augmenter unavailable: Skipping augmentation")
                # Still ensure expected directories for later stages
                self.dirs['final'].mkdir(parents=True, exist_ok=True)
                (self.dirs['final'] / 'pivot_pairs').mkdir(parents=True, exist_ok=True)
                span.set_attribute("augmented.total", 0)
                return
            
            for pair in self.config.data.augmentation_pairs:
                source, target = pair.split('-')
                
                # Backtranslation
                mono_file = self.dirs['raw'] / f"mono_{source}.txt"
                if mono_file.exists():
                    output_file = self.dirs['final'] / f"augmented_{pair}.txt"
                    
                    stats = self.augmenter.augment_with_backtranslation(
                        monolingual_file=str(mono_file),
                        source_lang=source,
                        target_lang=target,
                        output_file=str(output_file)
                    )
                    # stats is a dict; count augmented lines
                    if isinstance(stats, dict):
                        augmented_total += int(stats.get('augmented', 0))
            
            # Pivot translations
            try:
                pivot_stats = self.augmenter.generate_pivot_translations(
                    english_pairs_dir=str(self.dirs['sampled'])
                )
                if isinstance(pivot_stats, dict):
                    augmented_total += int(pivot_stats.get('total_pivot_pairs', 0))
            except Exception as e:
                self.logger.warning(f"Pivot generation skipped: {e}")
            
            span.set_attribute("augmented.total", augmented_total)
            self.logger.info(f"✅ Generated {augmented_total:,} synthetic samples")
    
    async def _create_training_ready(self):
        """Create training-ready data files"""
        with tracer.start_as_current_span("create_training_ready") as span:
            self.logger.info("📝 Creating training-ready data...")
            
            # Create monolingual corpora
            self.pipeline_connector.create_monolingual_corpora()
            
            # Create final training file
            self.pipeline_connector.create_final_training_file()
            
            # Update stats using known output path
            final_path = Path(self.config.data.processed_dir) / 'train_final.txt'
            if final_path.exists():
                self.state.total_size_gb = final_path.stat().st_size / (1024**3)
                
                with open(final_path, 'r', encoding='utf-8') as f:
                    self.state.total_sentences = sum(1 for _ in f)
            
            span.set_attribute("training_ready.sentences", self.state.total_sentences)
            span.set_attribute("training_ready.size_gb", self.state.total_size_gb)
    
    async def _validate_dataset(self):
        """Validate the final dataset"""
        with tracer.start_as_current_span("validate_dataset") as span:
            self.logger.info("✔️ Validating final dataset...")
            
            validations = {
                'size_check': self._validate_size(),
                'language_coverage': self._validate_languages(),
                'format_check': self._validate_format(),
                'quality_check': self._validate_quality()
            }
            
            all_valid = all(validations.values())
            
            if all_valid:
                self.logger.info("✅ All validations passed!")
            else:
                failed = [k for k, v in validations.items() if not v]
                self.logger.warning(f"⚠️ Failed validations: {failed}")
            
            span.set_attribute("validation.passed", all_valid)
            return all_valid
    
    async def _create_vocabulary(self):
        """Create vocabulary packs from processed data"""
        with tracer.start_as_current_span("create_vocabulary") as span:
            self.logger.info("📚 Creating vocabulary packs...")
            
            created_packs = self.vocab_connector.create_vocabularies_from_pipeline(
                processed_dir=str(self.dirs['processed'])
            )
            
            span.set_attribute("vocabulary.packs_created", len(created_packs))
            self.logger.info(f"✅ Created {len(created_packs)} vocabulary packs")
    
    # ============= VALIDATION METHODS =============
    
    def _validate_size(self) -> bool:
        """Validate dataset size meets requirements"""
        target_gb = self.config.data.total_size_gb
        actual_gb = self.state.total_size_gb
        
        if actual_gb < target_gb * 0.9:  # Allow 10% tolerance
            self.logger.warning(
                f"Dataset size ({actual_gb:.2f}GB) below target ({target_gb}GB)"
            )
            return False
        return True
    
    def _validate_languages(self) -> bool:
        """Validate language coverage"""
        required_languages = set(self.config.data.active_languages)
        found_languages = set()
        
        for file in self.dirs['processed'].glob('*_corpus.txt'):
            lang = file.stem.replace('_corpus', '')
            found_languages.add(lang)
        
        missing = required_languages - found_languages
        if missing:
            self.logger.warning(f"Missing languages: {missing}")
            return False
        return True
    
    def _validate_format(self) -> bool:
        """Validate data format"""
        # Check a sample of files
        for file in list(self.dirs['final'].glob('*.txt'))[:5]:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= 10:  # Check first 10 lines
                            break
                        if '\t' not in line and not line.strip():
                            self.logger.warning(f"Invalid format in {file.name}")
                            return False
            except Exception as e:
                self.logger.error(f"Cannot read {file.name}: {e}")
                return False
        return True
    
    def _validate_quality(self) -> bool:
        """Basic quality validation"""
        # Check if we have minimum sentences
        min_sentences = 100000
        if self.state.total_sentences < min_sentences:
            self.logger.warning(
                f"Too few sentences ({self.state.total_sentences} < {min_sentences})"
            )
            return False
        return True
    
    # ============= UTILITY METHODS =============
    
    def _generate_summary(self, execution_time: float) -> Dict[str, Any]:
        """Generate pipeline execution summary"""
        return {
            'execution_time_seconds': execution_time,
            'execution_time_hours': execution_time / 3600,
            'completed_stages': sum(1 for v in self.state.completed_stages.values() if v),
            'total_stages': len(self.state.completed_stages),
            'progress_percentage': self.state.get_progress(),
            'total_sentences': self.state.total_sentences,
            'total_size_gb': self.state.total_size_gb,
            'error_count': self.state.error_count,
            'resource_usage': resource_monitor.get_summary() if resource_monitor else {}
        }
    
    def _log_summary(self, summary: Dict[str, Any]):
        """Log pipeline execution summary"""
        self.logger.info("\n" + "="*60)
        self.logger.info("📊 PIPELINE SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Execution time: {summary['execution_time_hours']:.2f} hours")
        self.logger.info(f"Progress: {summary['progress_percentage']:.1f}%")
        self.logger.info(f"Total sentences: {summary['total_sentences']:,}")
        self.logger.info(f"Total size: {summary['total_size_gb']:.2f} GB")
        self.logger.info(f"Errors encountered: {summary['error_count']}")
        self.logger.info("="*60)
    
    def get_custom_sampler(self, 
                          dataset: Dataset,
                          sampler_type: str = 'temperature',
                          **kwargs) -> Sampler:
        """
        Get a custom sampler for the dataset.
        
        Args:
            dataset: The dataset to sample from
            sampler_type: Type of sampler ('temperature', 'balanced')
            **kwargs: Additional sampler arguments
            
        Returns:
            Configured sampler instance
        """
        if sampler_type == 'temperature':
            return TemperatureSampler(
                dataset=dataset,
                batch_size=kwargs.get('batch_size', 32),
                temperature=kwargs.get('temperature', 1.0),
                drop_last=kwargs.get('drop_last', False)
            )
        elif sampler_type == 'balanced':
            return BalancedLanguageSampler(
                dataset=dataset,
                batch_size=kwargs.get('batch_size', 32)
            )
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            'current_stage': self.state.current_stage.value if self.state.current_stage else None,
            'progress': self.state.get_progress(),
            'completed_stages': self.state.completed_stages,
            'total_sentences': self.state.total_sentences,
            'total_size_gb': self.state.total_size_gb,
            'is_complete': self.state.is_complete()
        }
    
    async def run_single_stage(self, stage: PipelineStage):
        """Run a single pipeline stage"""
        await self._execute_stage(stage)
    
    def reset_pipeline(self):
        """Reset pipeline state"""
        self.state = PipelineState(
            completed_stages={stage.value: False for stage in PipelineStage},
            current_stage=None
        )
        self._save_checkpoint()
        self.logger.info("Pipeline state reset")


def main():
    """Main entry point for the unified data pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified Data Pipeline')
    parser.add_argument('--config', default='config/base.yaml', help='Config file')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--stage', type=str, help='Run specific stage only')
    parser.add_argument('--reset', action='store_true', help='Reset pipeline state')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_dir="logs/data", log_level="INFO")
    
    # Load configuration
    from config.schemas import load_config
    config = load_config(args.config)
    
    # Create pipeline
    pipeline = UnifiedDataPipeline(config)
    
    # Handle reset
    if args.reset:
        pipeline.reset_pipeline()
        print("Pipeline state reset")
        return
    
    # Run pipeline
    if args.stage:
        # Run single stage
        stage = PipelineStage(args.stage)
        asyncio.run(pipeline.run_single_stage(stage))
    else:
        # Run full pipeline
        summary = asyncio.run(pipeline.run_pipeline(resume=args.resume))
        
        # Print summary
        print(f"\nPipeline completed in {summary['execution_time_hours']:.2f} hours")
        print(f"Total data: {summary['total_sentences']:,} sentences ({summary['total_size_gb']:.2f} GB)")


if __name__ == "__main__":
    main()