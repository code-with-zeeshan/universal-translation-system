import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from torch.utils.data import Sampler, Dataset
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from opentelemetry import trace

from config.schemas import RootConfig
from utils.exceptions import DataError
from utils.resource_monitor import resource_monitor
from utils.logging_config import setup_logging
from utils.common_utils import DirectoryManager

from data.unified_data_downloader import UnifiedDataDownloader, DatasetType
from data.smart_sampler import SmartDataSampler
from data.synthetic_augmentation import SyntheticDataAugmenter
from data.custom_samplers import TemperatureSampler, BalancedLanguageSampler
from data.pipeline_state import PipelineStage, PipelineState
from data.wikipedia_backtranslation import WikipediaBacktranslator
from data.knowledge_distillation import KnowledgeDistillator
from connector.pipeline_connector import PipelineConnector
from connector.vocabulary_connector import VocabularyConnector
from utils.constants import CONFIG_DIR, BASE_CONFIG_FILENAME, LOG_DIR

tracer = trace.get_tracer(__name__)


CORE_STAGES = [
    PipelineStage.DOWNLOAD_TRAIN,
    PipelineStage.SAMPLE_FILTER,
    PipelineStage.AUGMENT,
    PipelineStage.CREATE_READY,
    PipelineStage.COMET_QUALITY,
    PipelineStage.VALIDATE,
    PipelineStage.VOCABULARY,
]


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
            
            # Strategy 2: Wikipedia backtranslation data
            try:
                self.wikipedia = WikipediaBacktranslator()
            except Exception as e:
                self.logger.warning(f"Wikipedia backtranslator unavailable ({e})")
                self.wikipedia = None
            
            # Strategy 3: Knowledge distillation from NLLB-3.3B
            try:
                self.distillator = KnowledgeDistillator()
            except Exception as e:
                self.logger.warning(f"Knowledge distillator unavailable ({e})")
                self.distillator = None
            
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
            if stages is not None:
                stages_to_run = stages
            elif (hasattr(self.config, 'pipeline') and self.config.pipeline
                  and self.config.pipeline.enabled_stages):
                stages_to_run = [PipelineStage(s) for s in self.config.pipeline.enabled_stages]
            else:
                stages_to_run = CORE_STAGES
            
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
                    PipelineStage.WIKIPEDIA_BT: self._download_wikipedia_bt,
                    PipelineStage.DIRECT_OPUS: self._download_direct_opus,
                    PipelineStage.SAMPLE_FILTER: self._sample_and_filter_data,
                    PipelineStage.AUGMENT: self._augment_data,
                    PipelineStage.DISTILL: self._distill_data,
                    PipelineStage.CREATE_READY: self._create_training_ready,
                    PipelineStage.VALIDATE: self._validate_dataset,
                    PipelineStage.VOCABULARY: self._create_vocabulary,
                    PipelineStage.COMET_QUALITY: self._comet_quality_filter,
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
        """Download evaluation datasets or skip if already present"""
        with tracer.start_as_current_span("download_evaluation_data") as span:
            if getattr(self, 'dry_run', False):
                self.logger.info("🧪 Dry-run: Skipping evaluation downloads and using synthesized samples")
                span.set_attribute("dry_run", True)
                return

            # Skip if all expected eval files already exist
            eval_dir = self.dirs['base'] / 'evaluation'
            eval_pairs = self.downloader.get_required_pairs(DatasetType.EVALUATION)
            if eval_dir.exists():
                existing = sum(1 for p in eval_pairs if (eval_dir / f"{p.pair_string}.tsv").exists())
                if existing == len(eval_pairs):
                    self.logger.info(f"⏭️ All {existing} evaluation pair files exist, skipping download")
                    span.set_attribute("skipped", True)
                    return
                elif existing > 0:
                    self.logger.info(f"📥 {existing}/{len(eval_pairs)} eval files exist, downloading missing...")
            
            self.logger.info("📥 Downloading evaluation data...")
            stats = self.downloader.download_all(
                output_dir=str(self.dirs['base']),
                dataset_types=[DatasetType.EVALUATION]
            )
            span.set_attribute("eval_data.files", stats.get('eval_files', 0))
            self.logger.info(f"✅ Downloaded {stats.get('eval_files', 0)} evaluation datasets")
    
    async def _download_training_data(self):
        """Download training data based on strategy or synthesize in dry-run"""
        with tracer.start_as_current_span("download_training_data") as span:
            if getattr(self, 'dry_run', False):
                self.logger.info("🧪 Dry-run: Skipping training downloads and using synthesized samples")
                span.set_attribute("dry_run", True)
                return
            
            # Merge existing domain data before any skip check
            self._merge_domain_data()

            # Skip download only if ALL expected pair files already exist
            expected_pairs = list(self.config.data.training_distribution.keys())
            raw_dir = self.dirs['base'] / 'raw'
            if raw_dir.exists():
                existing = [p for p in expected_pairs if (raw_dir / f"{p}.txt").exists()]
                if len(existing) == len(expected_pairs):
                    self.logger.info(f"⏭️ All {len(expected_pairs)} pairs exist in {raw_dir}, skipping download")
                    span.set_attribute("skipped", True)
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

            # Merge domain-specific data into raw training files
            self._merge_domain_data()
    
    def _merge_domain_data(self):
        """Merge domain-specific data (medical/legal/tech) into raw pair files."""
        raw_dir = self.dirs['base'] / 'raw'
        for domain in ('medical', 'legal', 'tech'):
            domain_dir = raw_dir / domain
            if not domain_dir.exists():
                continue
            for fpath in domain_dir.glob('*_*.txt'):
                pair_str = fpath.stem.split('_')[0]
                target = raw_dir / f"{pair_str}.txt"
                if '-' not in pair_str:
                    continue
                count = 0
                with open(fpath, 'r', encoding='utf-8') as fin, \
                     open(target, 'a', encoding='utf-8') as fout:
                    for line in fin:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            fout.write(f"{parts[0]}\t{parts[1]}\n")
                            count += 1
                self.logger.info(f"Merged {count:,} {domain} samples into {target.name}")

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
        """Augment with synthetic data — false friends, idioms, tone, backtranslation"""
        with tracer.start_as_current_span("augment_data") as span:
            self.logger.info("🤖 Augmenting with synthetic data...")

            augmented_total = 0

            if self.augmenter is None or getattr(self, 'dry_run', False):
                self.logger.info("🧪 Dry-run or augmenter unavailable: Skipping augmentation")
                self.dirs['final'].mkdir(parents=True, exist_ok=True)
                (self.dirs['final'] / 'pivot_pairs').mkdir(parents=True, exist_ok=True)
                span.set_attribute("augmented.total", 0)
                return

            # 1) False friends + idioms (template and dynamic)
            from data.synthetic_augmentation import run_all_augmentations
            try:
                aug_results = run_all_augmentations(self.config, self.config.data.active_languages)
                for k, v in aug_results.items():
                    if isinstance(v, dict):
                        augmented_total += int(v.get('generated', 0))
            except Exception as e:
                self.logger.warning(f"False friend/idiom generation skipped: {e}")

            # 2) Backtranslation for each augmentation pair
            for pair in self.config.data.augmentation_pairs:
                source, target = pair.split('-')

                mono_file = self.dirs['raw'] / f"mono_{source}.txt"
                if mono_file.exists():
                    output_file = self.dirs['final'] / f"augmented_{pair}.txt"
                    stats = self.augmenter.augment_with_backtranslation(
                        monolingual_file=str(mono_file),
                        source_lang=source,
                        target_lang=target,
                        output_file=str(output_file)
                    )
                    if isinstance(stats, dict):
                        augmented_total += int(stats.get('augmented', 0))

            # 3) Pivot translations
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
    
    async def _download_wikipedia_bt(self):
        """Download Wikipedia monolingual data for backtranslation (strategy 2)."""
        with tracer.start_as_current_span("download_wikipedia_bt") as span:
            if getattr(self, 'dry_run', False) or self.wikipedia is None:
                self.logger.info("🧪 Dry-run or Wikipedia unavailable: Skipping")
                span.set_attribute("skipped", True)
                return

            self.logger.info("🌐 Downloading Wikipedia monolingual data...")
            self.wikipedia.output_dir = self.dirs['raw']
            langs = self.config.data.active_languages
            stats = self.wikipedia.download_monolingual(langs=langs, max_per_lang=200_000)
            for lang, count in stats.items():
                span.set_attribute(f"wikipedia.{lang}", count)
            self.logger.info(f"✅ Retrieved Wikipedia data for {sum(1 for c in stats.values() if c > 0)} languages")

    async def _download_direct_opus(self):
        """Direct OPUS.nlpl.eu download fallback for pairs missing after HF download (strategy 4)."""
        with tracer.start_as_current_span("download_direct_opus") as span:
            if getattr(self, 'dry_run', False):
                self.logger.info("🧪 Dry-run: Skipping direct OPUS")
                span.set_attribute("skipped", True)
                return

            raw_dir = self.dirs['base'] / 'raw'
            self.logger.info("📦 Direct OPUS download (fallback for missing pairs)...")

            schedule = self.downloader.get_download_schedule(DatasetType.TRAINING)
            downloaded = 0
            for batch in schedule:
                for pair in batch.get('pairs', []):
                    pair_str = pair.pair_string
                    if (raw_dir / f"{pair_str}.txt").exists():
                        continue
                    if pair_str not in self.required_pairs:
                        continue
                    ok = self.downloader._download_direct_opus(pair, raw_dir)
                    if ok:
                        downloaded += 1
            span.set_attribute("direct_opus.downloaded", downloaded)
            self.logger.info(f"✅ Direct OPUS: {downloaded} additional pairs downloaded")

    async def _distill_data(self):
        """Knowledge distillation from NLLB-3.3B teacher (strategy 3)."""
        with tracer.start_as_current_span("distill_data") as span:
            if getattr(self, 'dry_run', False) or self.distillator is None:
                self.logger.info("🧪 Dry-run or distillator unavailable: Skipping")
                span.set_attribute("skipped", True)
                return

            self.logger.info("🧪 Running knowledge distillation from NLLB-3.3B...")
            stats = self.distillator.distill_sampled_dir(
                str(self.dirs['sampled']),
                str(self.dirs['final'] / 'distilled'),
                max_pairs_per_pair=50_000,
            )
            for pair, count in stats.items():
                span.set_attribute(f"distilled.{pair}", count)
            self.logger.info(f"✅ Distilled {sum(stats.values()):,} pairs")

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
                processed_dir=str(self.dirs['processed']),
                output_dir=self.config.vocabulary.vocab_dir,
                vocab_size=self.config.vocabulary.vocab_size,
            )
            
            span.set_attribute("vocabulary.packs_created", len(created_packs))
            self.logger.info(f"✅ Created {len(created_packs)} vocabulary packs")
    
    # ============= COMET QUALITY FILTER =============

    async def _comet_quality_filter(self):
        """Filter training-ready data by COMET quality score.

        Reads the final training file, scores each pair with COMET,
        and writes a filtered version keeping only pairs above the threshold.
        """
        with tracer.start_as_current_span("comet_quality_filter") as span:
            from evaluation.evaluator import COMET_AVAILABLE

            if getattr(self, 'dry_run', False) or not COMET_AVAILABLE:
                self.logger.info("COMET not available — skipping quality filter")
                span.set_attribute("skipped", True)
                return

            final_path = Path(self.config.data.processed_dir) / 'train_final.txt'
            filtered_path = Path(self.config.data.processed_dir) / 'train_final_filtered.txt'
            if not final_path.exists():
                self.logger.warning(f"Final training file not found: {final_path}")
                return

            threshold = 0.7
            if hasattr(self.config, 'pipeline') and self.config.pipeline:
                threshold = self.config.pipeline.comet_quality_threshold

            try:
                from comet import download_model, load_from_checkpoint
                self.logger.info("Loading COMET model for quality filtering...")
                model_path = download_model("Unbabel/wmt22-comet-da")
                comet_model = load_from_checkpoint(model_path)

                pairs = []
                with open(final_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            pairs.append((parts[0], parts[1]))

                self.logger.info(f"Scoring {len(pairs):,} pairs with COMET...")
                comet_data = [{"src": s, "mt": t, "ref": t} for s, t in pairs]
                scores = comet_model.predict(comet_data, batch_size=64, gpus=1)

                kept = 0
                with open(filtered_path, 'w', encoding='utf-8') as f:
                    for (src, tgt), score in zip(pairs, scores.scores):
                        if score >= threshold:
                            f.write(f"{src}\t{tgt}\n")
                            kept += 1

                # Replace original with filtered
                filtered_path.replace(final_path)

                self.logger.info(
                    f"COMET filter: kept {kept:,}/{len(pairs):,} pairs "
                    f"(threshold={threshold:.2f})"
                )
                span.set_attribute("comet.total", len(pairs))
                span.set_attribute("comet.kept", kept)

            except Exception as e:
                self.logger.warning(f"COMET filtering failed: {e}")
                span.set_attribute("error", str(e))

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
    parser.add_argument('--config', default=f"{CONFIG_DIR}/{BASE_CONFIG_FILENAME}", help='Config file')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--stage', type=str, help='Run specific stage only')
    parser.add_argument('--reset', action='store_true', help='Reset pipeline state')
    parser.add_argument('--eval-only', action='store_true', help='Download evaluation data only, skip all other stages')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_dir=f"{LOG_DIR}/data", log_level="INFO")
    
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
    if args.eval_only:
        print("\n🧪 Running evaluation-only download...")
        stage = PipelineStage.DOWNLOAD_EVAL
        asyncio.run(pipeline.run_single_stage(stage))
        print("✅ Evaluation data download complete")
    elif args.stage:
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
