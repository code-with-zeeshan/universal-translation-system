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
from utils.common_utils import RuntimeDirectoryManager, DirectoryManager

from pipeline.data.downloader import UnifiedDataDownloader, DatasetType, LanguagePair, DownloadPriority
from pipeline.data.sampler import SmartDataSampler
from pipeline.data.augmentation import SyntheticDataAugmenter
from pipeline.training.samplers import TemperatureSampler, BalancedLanguageSampler
from pipeline.data.state import PipelineStage, PipelineState, _hash_config_section, STAGE_ORDER
from pipeline.data.backtranslation import WikipediaBacktranslator
from pipeline.data.distillation import KnowledgeDistillator
from pipeline.connectors.data import PipelineConnector
from pipeline.connectors.vocabulary import VocabularyConnector
from pipeline.vocabulary.evolve import VocabularyEvolver
from utils.constants import CONFIG_DIR, BASE_CONFIG_FILENAME, TRAIN_FINAL_FILENAME

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


def _maybe_upload_to_hub(config, runtime_dirs):
    """Upload processed data+vocab to HF Hub if configured."""
    hub_cfg = getattr(config, 'hub', None)
    if not hub_cfg or not hub_cfg.dataset_repo_id or not hub_cfg.auto_upload:
        return
    try:
        from pipeline.data.hub_sync import upload_processed_data
        processed_dir = Path(runtime_dirs.processed_dir) if hasattr(runtime_dirs, 'processed_dir') else Path(config.data.processed_dir)
        vocab_dir = Path(runtime_dirs.vocab_dir) if hasattr(runtime_dirs, 'vocab_dir') else Path(config.data.output_dir).parent / "vocabulary" / "vocab"
        datasets_dir = Path(runtime_dirs.datasets_dir) if hasattr(runtime_dirs, 'datasets_dir') else processed_dir
        n = upload_processed_data(hub_cfg.dataset_repo_id, processed_dir, vocab_dir, token=hub_cfg.token, datasets_dir=datasets_dir)
        logger.info("Uploaded %d files to HF Hub dataset %s", n, hub_cfg.dataset_repo_id)
    except Exception as e:
        logger.warning("HF Hub upload failed (non-fatal): %s", e)


def ensure_data_ready(config: 'RootConfig') -> bool:
    """Ensure data (train_final.txt, val_final.txt) and vocab are available.

    Tries in order:
      1. Local check — if both present, return immediately.
      2. Download from HF Hub (if hub.dataset_repo_id set).
      3. Run the data pipeline for whichever components are still missing.

    Returns True when all required data and vocab are available.
    """
    from pipeline.data.hub_sync import download_processed_data, whats_missing

    _rdm = RuntimeDirectoryManager(config=config)
    processed_dir = _rdm.processed_dir
    vocab_dir = _rdm.vocab_dir
    datasets_dir = _rdm.datasets_dir

    missing = whats_missing(processed_dir, vocab_dir, datasets_dir)
    if not missing.get("data") and not missing.get("vocab"):
        logger.info("All data and vocab already present locally")
        return True

    # Step 2 — download from HF Hub if configured
    hub_cfg = getattr(config, 'hub', None)
    if hub_cfg and hub_cfg.dataset_repo_id:
        logger.info("Trying download from HF Hub %s ...", hub_cfg.dataset_repo_id)
        download_processed_data(
            hub_cfg.dataset_repo_id, processed_dir, vocab_dir,
            token=hub_cfg.token, datasets_dir=datasets_dir,
        )

    # Step 3 — re-check what's still missing
    missing = whats_missing(processed_dir, vocab_dir, datasets_dir)
    if not missing.get("data") and not missing.get("vocab"):
        logger.info("All data and vocab ready after download")
        return True

    # Step 4 — run pipeline for missing components
    stages_to_run: list[PipelineStage] = []
    if missing.get("data"):
        logger.info("Data still missing — running data pipeline stages")
        stages_to_run.extend(CORE_STAGES)
    if missing.get("vocab"):
        logger.info("Vocab still missing — running vocabulary stage")
        if PipelineStage.VOCABULARY not in stages_to_run:
            stages_to_run.append(PipelineStage.VOCABULARY)

    if not stages_to_run:
        return True

    pipeline = UnifiedDataPipeline(config)
    import asyncio
    asyncio.run(pipeline.run_pipeline(resume=True, stages=stages_to_run))

    # Final verification
    missing = whats_missing(processed_dir, vocab_dir, datasets_dir)
    if missing.get("data") or missing.get("vocab"):
        logger.error("Still missing after pipeline run: %s", missing)
        return False
    return True


CORE_STAGES = [
    PipelineStage.DOWNLOAD_TRAIN,
    PipelineStage.SAMPLE_FILTER,
    PipelineStage.AUGMENT,
    PipelineStage.CREATE_READY,
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
            
            # Directory structure: NOT created eagerly here. Created on first
            # call to run_pipeline() so empty dirs don't appear before operations.
            self.runtime_dirs = RuntimeDirectoryManager(config=self.config)
            self.dirs = {}
            
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
            
            # Pre-compute config sections for fingerprinting
            self._config_sections = self._build_config_sections()
            
            span.set_attribute("pipeline.initialized", True)
            self.logger.info("✅ Unified data pipeline initialized")
    
    def _initialize_components(self):
        """Initialize all pipeline components with error handling"""
        try:
            # Use unified components
            self.downloader = UnifiedDataDownloader(self.config)
            self.sampler = SmartDataSampler()

            # All NLLB stages use the same distilled 1.3B model.
            # This avoids loading two separate NLLB variants and reduces
            # GPU memory requirements by ~5x vs. the 3.3B model.
            NLLB_MODEL = "facebook/nllb-200-distilled-1.3B"

            # Only instantiate augmenter if heavy deps are available; otherwise, create a stub that skips
            try:
                self.augmenter = SyntheticDataAugmenter(self.config, base_model=NLLB_MODEL)
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
            
            # Wikipedia backtranslator — used inside _augment_data to download
            # monolingual data for backtranslation sources (no dedicated stage needed).
            try:
                self.wikipedia = WikipediaBacktranslator()
            except Exception as e:
                self.logger.warning(f"Wikipedia backtranslator unavailable ({e})")
                self.wikipedia = None
            
            # Knowledge distillation from NLLB-1.3B (same model as augment)
            try:
                self.distillator = KnowledgeDistillator(model_name=NLLB_MODEL)
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
        checkpoint_path = self.runtime_dirs.data_dir / 'pipeline_checkpoint.json'
        
        with open(checkpoint_path, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2)
        
        self.logger.debug(f"Saved checkpoint: {self.state.get_progress():.1f}% complete")
    
    def _load_checkpoint(self):
        """Load pipeline state if checkpoint exists"""
        checkpoint_path = self.runtime_dirs.data_dir / 'pipeline_checkpoint.json'
        
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r') as f:
                    checkpoint_data = json.load(f)
                
                self.state = PipelineState.from_dict(checkpoint_data)
                
                self.logger.info(
                    f"Loaded checkpoint: {self.state.get_progress():.1f}% complete"
                    f"{' (pipeline fully completed)' if self.state.pipeline_complete else ''}"
                )
                
            except Exception as e:
                self.logger.warning(f"Could not load checkpoint: {e}")
    
    def _build_config_sections(self) -> Dict[str, object]:
        """Extract relevant config sections per stage for fingerprinting."""
        c = self.config
        sections = {}
        sections['download_evaluation'] = {'eval_pairs': list(getattr(c.data, 'evaluation_pairs', []))}
        sections['download_training'] = {
            'training_distribution': dict(c.data.training_distribution),
            'domain_data': list(getattr(c.data, 'domain_data', [])),
        }
        sections['sample_filter'] = {'training_distribution': dict(c.data.training_distribution)}
        sections['augment'] = {
            'augmentation_pairs': list(c.data.augmentation_pairs),
            'active_languages': list(c.data.active_languages),
            'high_resource_threshold': c.pipeline.high_resource_threshold if c.pipeline else 100_000,
        }
        sections['create_ready'] = {'train_val_split': getattr(c.data, 'train_val_split', 0.9)}
        sections['validate'] = {
            'total_size_gb': c.data.total_size_gb,
            'active_languages': list(c.data.active_languages),
        }
        sections['vocabulary'] = {
            'vocab_size': c.vocabulary.vocab_size,
            'vocab_dir': c.vocabulary.vocab_dir,
        }
        sections['comet_quality'] = {
            'comet_threshold': c.pipeline.comet_quality_threshold if c.pipeline else 0.7,
        }
        sections['knowledge_distillation'] = {'max_pairs_per_pair': 50_000}
        sections['direct_opus'] = {'training_distribution': list(c.data.training_distribution.keys())}
        return sections

    def _compute_stage_hash(self, stage: PipelineStage) -> str:
        """Hash the config section relevant to a pipeline stage."""
        section = self._config_sections.get(stage.value, {})
        return _hash_config_section(section)

    def _validate_stage_fingerprint(self, stage: PipelineStage) -> bool:
        """Check if a completed stage's input config has changed since it ran."""
        stored = self.state.stage_input_hashes.get(stage.value)
        if stored is None:
            return False
        return stored == self._compute_stage_hash(stage)

    def _verify_stage_outputs(self, stage: PipelineStage) -> bool:
        """Verify that a stage's actual output files exist on disk.

        Prevents skipping stages whose checkpoint metadata says 'done'
        but whose artifacts were deleted or never created.
        """
        _rdm = self.runtime_dirs
        VERIFIERS = {
            PipelineStage.VOCABULARY: lambda _: (
                _rdm.vocab_dir.is_dir()
                and bool(list(_rdm.vocab_dir.glob("*_v*.msgpack")))
            ),
            PipelineStage.DOWNLOAD_TRAIN: lambda _: (
                (_rdm.processed_dir / "corpus").is_dir()
                and any((_rdm.processed_dir / "corpus").glob("*_corpus.txt"))
            ),
            PipelineStage.SAMPLE_FILTER: lambda _: (
                (_rdm.processed_dir / "sampled").is_dir()
                and any((_rdm.processed_dir / "sampled").glob("*_sampled.txt"))
            ),
            PipelineStage.AUGMENT: lambda _: (
                (_rdm.processed_dir / "augment").is_dir()
            ),
            PipelineStage.CREATE_READY: lambda _: (
                (_rdm.train_final_path).is_file()
            ),
            PipelineStage.VALIDATE: lambda _: True,
            PipelineStage.COMET_QUALITY: lambda _: True,
        }
        check = VERIFIERS.get(stage)
        if check is None:
            return True
        return check(None)

    # ============= MAIN PIPELINE EXECUTION =============
    
    async def run_pipeline(self, 
                          resume: bool = True,
                          force: bool = False,
                          stages: Optional[List[PipelineStage]] = None) -> Dict[str, Any]:
        """
        Execute the complete data pipeline.
        
        Args:
            resume: Whether to resume from checkpoint (default: True, auto-detect)
            force: If True, ignore checkpoint and re-run all stages
            stages: Specific stages to run (None = all)
            
        Returns:
            Pipeline execution summary
        """
        with tracer.start_as_current_span("UnifiedDataPipeline.run_pipeline") as span:
            span.set_attribute("resume", resume)
            span.set_attribute("force", force)
            span.set_attribute("dry_run", getattr(self, 'dry_run', False))

            # Create data directories now — not in __init__ — so empty dirs
            # don't appear before a pipeline is actually launched.
            self.dirs = self.runtime_dirs.ensure_data_structure()
            
            # If pipeline fully completed and not forced, short-circuit
            if self.state.pipeline_complete and not force and resume:
                self.logger.info("✅ Pipeline already fully completed. Use --force to re-run from scratch.")
                return self._generate_summary(0)
            
            if force:
                self.logger.info("🔁 Force mode: re-running all stages from scratch")
                self.state = PipelineState(
                    completed_stages={stage.value: False for stage in PipelineStage},
                    current_stage=None
                )
                self._save_checkpoint()
            
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
                stage_completed = self.state.completed_stages.get(stage.value, False)
                if stage_completed and not self._verify_stage_outputs(stage):
                    self.logger.warning(
                        f"🔄 Checkpoint marks '{stage.value}' as complete "
                        f"but output files are missing — re-running"
                    )
                    stage_completed = False
                
                if not resume or not stage_completed:
                    await self._execute_stage(stage)
                elif stage_completed and not self._validate_stage_fingerprint(stage):
                    self.logger.warning(
                        f"⚡ Config changed for '{stage.value}' — re-running it "
                        f"and invalidating downstream stages"
                    )
                    self.state.invalidate_downstream(stage)
                    await self._execute_stage(stage)
                else:
                    self.logger.info(f"⏩ Skipping {stage.value} (already completed, config unchanged)")
            
            # Clean up intermediate files once create_ready is done
            if self.state.completed_stages.get('create_ready', False):
                self._cleanup_intermediate_files()

            # Mark pipeline fully complete when all stages pass
            if self.state.is_complete():
                self.state.pipeline_complete = True
                self._save_checkpoint()
            
            # Calculate execution time
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Generate summary
            summary = self._generate_summary(execution_time)
            
            span.set_attribute("pipeline.completed", True)
            span.set_attribute("pipeline.execution_time", execution_time)
            
            if self.state.pipeline_complete:
                self.logger.info("✅ Pipeline fully completed!")
            else:
                self.logger.info(f"📌 Pipeline progress: {self.state.get_progress():.1f}%")
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
        corpus_dir = processed_dir / "corpus"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        for lang, lines in [('en', ['hello world', 'this is a test']),
                            ('es', ['hola mundo', 'esto es una prueba'])]:
            corp = corpus_dir / f"{lang}_corpus.txt"
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
                    self.state.stage_input_hashes[stage.value] = self._compute_stage_hash(stage)
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

            expected_pairs = list(self.config.data.training_distribution.keys())
            raw_dir = self.dirs['base'] / 'raw'

            # Check per-pair completion to skip already-downloaded pairs
            pairs_to_download = []
            for p in expected_pairs:
                if self.state.is_sub_stage_done('download_training', p):
                    self.logger.info(f"⏩ Already downloaded: {p}")
                elif (raw_dir / f"{p}.txt").exists():
                    self.state.mark_sub_stage('download_training', p)
                    self.logger.info(f"⏩ Already exists: {p}")
                else:
                    pairs_to_download.append(p)

            if not pairs_to_download:
                self.logger.info(f"⏭️ All {len(expected_pairs)} pairs already downloaded")
                span.set_attribute("skipped", True)
                return

            self.logger.info(f"📥 Downloading {len(pairs_to_download)} missing pairs...")
            
            # Use strategy-based download
            schedule = self.downloader.get_download_schedule(DatasetType.TRAINING)
            for batch in schedule:
                self.logger.info(f"Processing batch: {batch['batch_name']}")
            
            stats = self.downloader.download_all(
                output_dir=str(self.dirs['base']),
                dataset_types=[DatasetType.TRAINING]
            )
            
            # Auto-fallback: try direct OPUS for any pair that still has no data.
            # This runs unconditionally inside the download_training stage, making
            # the dedicated 'direct_opus' pipeline stage redundant. No need to
            # enable it separately — missing pairs get this retry automatically.
            for p_str in expected_pairs:
                if not (raw_dir / f"{p_str}.txt").exists():
                    src, tgt = p_str.split('-', 1)
                    fallback_pair = LanguagePair(
                        source=src, target=tgt,
                        priority=DownloadPriority.MEDIUM,
                        expected_size=self.config.data.training_distribution.get(p_str, 50000),
                        data_sources=['opus_direct'],
                    )
                    self.logger.info(f"🔄 Auto-fallback: trying direct OPUS for {p_str}...")
                    ok = self.downloader._download_direct_opus(fallback_pair, raw_dir)
                    if ok:
                        self.logger.info(f"✅ Direct OPUS fallback succeeded for {p_str}")
            
            # Mark each pair as done after successful download
            for p in expected_pairs:
                self.state.mark_sub_stage('download_training', p)
            self._save_checkpoint()
            
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
                if self.state.is_sub_stage_done('sample_filter', pair_str):
                    self.logger.info(f"⏩ Already sampled: {pair_str}")
                    continue
                    
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
                    self.state.mark_sub_stage('sample_filter', pair_str)
                    self._save_checkpoint()
                    self.logger.info(f"✅ Sampled {pair_str}: {stats['written_count']:,} sentences")
            
            self.state.total_sentences += total_sampled
            span.set_attribute("sampled.total", total_sampled)

            # Auto-evolve vocab for low-resource pairs (<50% of target size)
            low_resource_pairs = []
            for pair_str, target_size in distribution.items():
                sampled_file = self.dirs['sampled'] / f"{pair_str}_sampled.txt"
                if sampled_file.exists():
                    line_count = 0
                    with open(sampled_file) as _f:
                        for _ in _f:
                            line_count += 1
                    if line_count < target_size * 0.5:
                        low_resource_pairs.append((pair_str, line_count, target_size))

            if low_resource_pairs:
                for pair_str, actual, target in low_resource_pairs:
                    self.logger.warning(
                        f"⚠️ Low-resource pair {pair_str}: {actual:,}/{target:,} "
                        f"({actual/target*100:.0f}% of target) — triggering vocab evolution"
                    )
                try:
                    evolver = VocabularyEvolver(vocab_dir=str(self.dirs.get('vocab', self.dirs['base'] / 'vocab')))
                    results = evolver.evolve_all_packs()
                    if results:
                        self.logger.info(f"✅ Evolved {len(results)} vocab packs with new subwords: {results}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Vocab evolution skipped (no existing packs yet): {e}")

    async def _augment_data(self):
        """Augment with synthetic data — false friends, idioms, equivalence pairs, backtranslation.

        Wikipedia monolingual data is downloaded here (CPU-only, streaming) to feed
        the backtranslation step. This replaces the former separate WIKIPEDIA_BT stage.
        """
        with tracer.start_as_current_span("augment_data") as span:
            self.logger.info("🤖 Augmenting with synthetic data...")

            augmented_total = 0

            if self.augmenter is None or getattr(self, 'dry_run', False):
                self.logger.info("🧪 Dry-run or augmenter unavailable: Skipping augmentation")
                self.dirs['final'].mkdir(parents=True, exist_ok=True)
                (self.dirs['final'] / 'pivot_pairs').mkdir(parents=True, exist_ok=True)
                span.set_attribute("augmented.total", 0)
                return

            # 0) Download Wikipedia monolingual data as backtranslation source (CPU-only)
            if self.wikipedia is not None:
                try:
                    self.logger.info("🌐 Downloading Wikipedia monolingual data for backtranslation...")
                    self.wikipedia.output_dir = self.dirs['raw']
                    wiki_stats = self.wikipedia.download_monolingual(
                        langs=self.config.data.active_languages,
                        max_per_lang=200_000,
                    )
                    for lang, count in wiki_stats.items():
                        span.set_attribute(f"wikipedia.{lang}", count)
                    self.logger.info(
                        f"✅ Wikipedia: {sum(1 for c in wiki_stats.values() if c > 0)} languages available"
                    )
                except Exception as e:
                    self.logger.warning(f"Wikipedia download failed (non-fatal): {e}")

            # 1) False friends + idioms (template + dynamic + equivalence)
            from pipeline.data.augmentation import run_all_augmentations
            try:
                aug_results = run_all_augmentations(self.config, self.config.data.active_languages)
                for k, v in aug_results.items():
                    if isinstance(v, dict):
                        augmented_total += int(v.get('generated', 0))
            except Exception as e:
                self.logger.warning(f"False friend/idiom generation skipped: {e}")

            # 2) Backtranslation for each augmentation pair
            #    Skip high-resource pairs that already have sufficient OPUS/CCMatrix data.
            # Effectively disable the threshold — run NLLB backtranslation on all pairs
            # regardless of existing OPUS/CCMatrix data volume. Config can override.
            hr_threshold = 100_000_000
            if hasattr(self.config, 'pipeline') and self.config.pipeline:
                hr_threshold = self.config.pipeline.high_resource_threshold

            for pair in self.config.data.augmentation_pairs:
                source, target = pair.split('-')

                # Skip if this pair has enough data already (high-resource threshold)
                existing_target = self.config.data.training_distribution.get(pair, 0)
                sampled_file = self.dirs['sampled'] / f"{pair}_sampled.txt"
                existing_count = 0
                if sampled_file.exists():
                    with open(sampled_file) as _f:
                        for _ in _f:
                            existing_count += 1

                if existing_count >= hr_threshold or existing_target >= hr_threshold:
                    self.logger.info(
                        f"⏭️ Skipping NLLB backtranslation for {pair}: "
                        f"{max(existing_count, existing_target):,} existing/examples >= "
                        f"high_resource_threshold={hr_threshold}"
                    )
                    span.set_attribute(f"bt_skipped.{pair}", True)
                    continue

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
                    english_pairs_dir=str(self.dirs['sampled']),
                    output_dir=str(self.dirs['final'] / 'pivot_pairs')
                )
                if isinstance(pivot_stats, dict):
                    augmented_total += int(pivot_stats.get('total_pivot_pairs', 0))
            except Exception as e:
                self.logger.warning(f"Pivot generation skipped: {e}")

            span.set_attribute("augmented.total", augmented_total)
            self.logger.info(f"✅ Generated {augmented_total:,} synthetic samples")
    
    # NOTE: Wikipedia download is now part of _augment_data (merged).
    # The former _download_wikipedia_bt method was removed to avoid having
    # a dedicated stage for a cheap streaming download.

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

            self.logger.info("🧪 Running knowledge distillation from NLLB-1.3B...")
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
            final_path = self.runtime_dirs.train_final_path
            if final_path.exists():
                self.state.total_size_gb = final_path.stat().st_size / (1024**3)
                
                with open(final_path, 'r', encoding='utf-8') as f:
                    self.state.total_sentences = sum(1 for _ in f)
            
            span.set_attribute("training_ready.sentences", self.state.total_sentences)
            span.set_attribute("training_ready.size_gb", self.state.total_size_gb)
    
    def _cleanup_intermediate_files(self):
        """Remove raw/ and sampled/ directories after training data is ready."""
        for subdir in ('raw', 'sampled'):
            d = self.dirs['base'] / subdir
            if d.exists() and d.is_dir():
                import shutil
                shutil.rmtree(d)
                self.logger.info(f"🧹 Cleaned up intermediate directory: {d}")

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
                output_dir=self.runtime_dirs.vocab_dir,
                vocab_size=self.config.vocabulary.vocab_size,
                max_model_vocab_size=self.config.model.vocab_size,
            )
            
            span.set_attribute("vocabulary.packs_created", len(created_packs))
            self.logger.info(f"✅ Created {len(created_packs)} vocabulary packs")
    
    # ============= COMET QUALITY FILTER =============

    def _comet_filter_file(
        self, path: Path, comet_model: Any, threshold: float
    ) -> int:
        """Score one file with COMET and rewrite it in-place, keeping only
        pairs above *threshold*. Preserves the full 4-column line format."""
        if not path.exists():
            self.logger.warning(f"COMET: file not found, skipping: {path}")
            return 0

        lines: list[str] = []
        with open(path, 'r', encoding='utf-8') as f:
            lines = [line.rstrip('\n') for line in f]

        if not lines:
            self.logger.info(f"COMET: {path.name} is empty, skipping")
            return 0

        self.logger.info(f"Scoring {len(lines):,} pairs from {path.name} with COMET...")
        pairs = [line.split('\t') for line in lines]
        comet_data = [{"src": p[0], "mt": p[1], "ref": p[1]} for p in pairs]
        scores = comet_model.predict(comet_data, batch_size=64, gpus=1)

        kept_lines: list[str] = []
        for line, score in zip(lines, scores.scores):
            if score >= threshold:
                kept_lines.append(line)

        with open(path, 'w', encoding='utf-8') as f:
            for line in kept_lines:
                f.write(line + '\n')

        self.logger.info(
            f"COMET {path.name}: kept {len(kept_lines):,}/{len(lines):,} "
            f"(threshold={threshold:.2f})"
        )
        return len(kept_lines)

    async def _comet_quality_filter(self):
        """Filter training-ready data by COMET quality score.

        Scores every source→target pair with COMET and drops pairs below
        the quality threshold. Preserves the full 4-column row format.
        Filters both train_final.txt and val_final.txt for consistency.
        """
        with tracer.start_as_current_span("comet_quality_filter") as span:
            from evaluation.evaluator import COMET_AVAILABLE

            if getattr(self, 'dry_run', False) or not COMET_AVAILABLE:
                self.logger.info("COMET not available — skipping quality filter")
                span.set_attribute("skipped", True)
                return

            datasets = self.runtime_dirs.datasets_dir
            datasets.mkdir(parents=True, exist_ok=True)
            train_path = datasets / 'train_final.txt'
            val_path = datasets / 'val_final.txt'
            if not train_path.exists():
                self.logger.warning(f"Training file not found: {train_path}")
                return

            threshold = 0.7
            if hasattr(self.config, 'pipeline') and self.config.pipeline:
                threshold = self.config.pipeline.comet_quality_threshold

            try:
                from comet import download_model, load_from_checkpoint
                self.logger.info("Loading COMET model for quality filtering...")
                model_path = download_model("Unbabel/wmt22-comet-da")
                comet_model = load_from_checkpoint(model_path)

                train_kept = self._comet_filter_file(train_path, comet_model, threshold)
                val_kept = self._comet_filter_file(val_path, comet_model, threshold)

                span.set_attribute("comet.train_kept", train_kept)
                span.set_attribute("comet.val_kept", val_kept)

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
        
        for file in self.dirs['corpus'].glob('*_corpus.txt'):
            lang = file.stem.replace('_corpus', '')
            found_languages.add(lang)
        
        missing = required_languages - found_languages
        if missing:
            self.logger.warning(f"Missing languages: {missing}")
            return False
        return True
    
    def _validate_format(self) -> bool:
        """Validate format of final training/validation files and augmented sources."""
        targets = [
            ("train_final.txt", self.runtime_dirs.train_final_path),
            ("val_final.txt", self.runtime_dirs.val_final_path),
        ]
        for name, path in targets:
            if not path.exists():
                self.logger.error(f"Missing final file: {path}")
                return False
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= 50:
                            break
                        if '\t' not in line:
                            self.logger.warning(f"Invalid format in {name} line {i + 1}")
                            return False
                        parts = line.strip().split('\t')
                        if len(parts) < 4:
                            self.logger.warning(f"Expected 4+ columns in {name} line {i + 1}, got {len(parts)}")
                            return False
            except Exception as e:
                self.logger.error(f"Cannot read {name}: {e}")
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
        """Reset pipeline state — clears completion, sub-progress, fingerprints, and complete flag."""
        self.state = PipelineState(
            completed_stages={stage.value: False for stage in PipelineStage},
            current_stage=None
        )
        self._save_checkpoint()
        self.logger.info("🔄 Pipeline state fully reset")


def main():
    """Main entry point for the unified data pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified Data Pipeline')
    parser.add_argument('--config', default=f"{CONFIG_DIR}/{BASE_CONFIG_FILENAME}", help='Config file')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume from checkpoint (default: auto-detect)')
    parser.add_argument('--no-resume', action='store_false', dest='resume',
                        help='Ignore checkpoint and run all stages')
    parser.add_argument('--force', action='store_true',
                        help='Clear checkpoint and re-run from scratch')
    parser.add_argument('--stage', type=str, help='Run specific stage only')
    parser.add_argument('--reset', action='store_true', help='Reset pipeline state')
    parser.add_argument('--eval-only', action='store_true', help='Download evaluation data only, skip all other stages')
    parser.add_argument('--download-max-workers', type=int, default=None,
                        help='Override max parallel downloads per batch')
    parser.add_argument('--download-parallel-batches', action='store_true',
                        help='Enable parallel batch downloads')
    parser.add_argument('--datasets-cache-dir', type=str, default=None,
                        help='HuggingFace datasets cache directory (default: HF default cache)')
    parser.add_argument('--hub-repo-id', type=str, default=None,
                        help='HF Hub repo ID to upload processed data+vocab after pipeline')

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level="INFO")

    # Load configuration
    from config.schemas import load_config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.download_max_workers is not None:
        config.data.download_max_workers = args.download_max_workers
    if args.download_parallel_batches:
        config.data.download_parallel_batches = True
    if args.datasets_cache_dir is not None:
        config.data.datasets_cache_dir = args.datasets_cache_dir
    if args.hub_repo_id is not None:
        from config.schemas import HubConfig
        config.hub = config.hub or HubConfig()
        config.hub.dataset_repo_id = args.hub_repo_id
        config.hub.auto_upload = True
    
    # Create pipeline
    pipeline = UnifiedDataPipeline(config)
    
    if args.reset:
        pipeline.reset_pipeline()
        logger.info("Pipeline state reset")
        return
    
    if args.force:
        pipeline.reset_pipeline()
        logger.info("Force mode: checkpoint cleared, starting fresh")
    
    if args.eval_only:
        logger.info("Running evaluation-only download...")
        stage = PipelineStage.DOWNLOAD_EVAL
        asyncio.run(pipeline.run_single_stage(stage))
        logger.info("Evaluation data download complete")
    elif args.stage:
        stage = PipelineStage(args.stage)
        asyncio.run(pipeline.run_single_stage(stage))
    else:
        summary = asyncio.run(pipeline.run_pipeline(resume=args.resume, force=args.force))
        if summary['execution_time_hours'] > 0:
            logger.info(f"Pipeline completed in {summary['execution_time_hours']:.2f} hours — "
                        f"{summary['total_sentences']:,} sentences ({summary['total_size_gb']:.2f} GB)")
        _maybe_upload_to_hub(pipeline.config, pipeline.runtime_dirs)


if __name__ == "__main__":
    main()
