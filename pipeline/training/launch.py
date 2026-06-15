from utils.common_utils import RuntimeDirectoryManager
# pipeline/training/launch.py
"""
Launch utilities for intelligent training system
Handles CLI, distributed setup, and orchestration
"""

import argparse
import sys
import os

# Set main PID before any imports, so spawned DataLoader workers can detect they
# are not the main process and suppress duplicated logging.
# setdefault prevents spawned children (which re-import this module) from
# overwriting the parent's PID.
os.environ.setdefault('OP_MAIN_PID', str(os.getpid()))
import torch
import torch.multiprocessing as mp
from pathlib import Path
import logging
import json
import yaml
from typing import Optional, Dict, Any, Tuple
import time
from datetime import datetime

from pipeline.training.trainer import IntelligentTrainer, train_intelligent
from runtime.encoder.universal_encoder import UniversalEncoder
from runtime.cloud_decoder import OptimizedUniversalDecoder
from pipeline.training.datasets import ModernParallelDataset
from utils.shutdown_handler import GracefulShutdown
from utils.model_versioning import ModelVersion
from utils.resource_monitor import resource_monitor
from utils.logging_config import setup_logging
from config.schemas import RootConfig, load_config as load_pydantic_config
from utils.constants import TRAIN_FINAL_FILENAME, VAL_FINAL_FILENAME, BEST_MODEL_FILENAME, TEST_FINAL_FILENAME, EVALUATION_REPORT_FILENAME
from utils.pipeline_checkpoint import PhaseCheckpoint, mark_stage_complete, is_stage_complete, invalidate_downstream, hash_config

# Centralized logging for training
setup_logging(log_dir=str(RuntimeDirectoryManager().logs_dir), log_level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("training")


def setup_environment():
    """Setup environment variables and optimizations"""
    # Distributed training environment
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12355'
    
    # PyTorch optimizations
    os.environ['TORCH_COMPILE_DEBUG'] = '0'
    os.environ['TORCH_LOGS'] = '+dynamo'
    os.environ['TORCHDYNAMO_DISABLE_CACHE_LIMIT'] = '1'
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    
    # CUDA optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def load_configuration_dynamic_or_yaml(config_path: str, dynamic: bool) -> RootConfig:
    """Load configuration from YAML or build a dynamic default when requested."""
    if dynamic or (config_path and config_path.strip().lower() == 'dynamic'):
        # Build a default RootConfig; trainer will refine based on hardware
        from config.schemas import DataConfig, ModelConfig, TrainingConfig, MemoryConfig, VocabularyConfig
        cfg = RootConfig(
            data=DataConfig(training_distribution={}),
            model=ModelConfig(),
            training=TrainingConfig(),
            memory=MemoryConfig(),
            vocabulary=VocabularyConfig()
        )
        logger.info("✅ Using dynamic configuration (no YAML)")
        return cfg
    # Fallback: load YAML
    try:
        config = load_pydantic_config(config_path)
        logger.info(f"✅ Configuration loaded from: {config_path}")
        return config
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        sys.exit(1)


def initialize_models(config: RootConfig) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Initialize encoder and decoder models, bootstrapping from pretrained if needed"""
    logger.info("🔧 Initializing models...")
    
    encoder_path = RuntimeDirectoryManager().encoder_models_dir / "universal_encoder_initial.pt"
    decoder_path = RuntimeDirectoryManager().decoder_models_dir / "universal_decoder_initial.pt"
    
    # Bootstrap from pretrained if weight files don't exist
    if not encoder_path.exists() or not decoder_path.exists():
        logger.info("No pretrained weights found. Bootstrapping from HuggingFace...")
        try:
            from pipeline.training.bootstrap import PretrainedModelBootstrapper
            bootstrapper = PretrainedModelBootstrapper(device="auto")
            
            if not encoder_path.exists():
                logger.info("Bootstrapping encoder from xlm-roberta-base...")
                bootstrapper.create_encoder_from_pretrained(
                    output_path=str(encoder_path),
                    target_hidden_dim=config.model.hidden_dim
                )
            
            if not decoder_path.exists():
                logger.info("Bootstrapping decoder from facebook/mbart-large-50...")
                bootstrapper.create_decoder_from_mbart(
                    output_path=str(decoder_path),
                    encoder_dim=config.model.hidden_dim,
                    decoder_dim=config.model.decoder_dim,
                    max_seq_length=config.model.max_seq_length,
                )
        except Exception as e:
            logger.warning(f"Bootstrap failed ({e}), training from random initialization")
    
    encoder = UniversalEncoder(
        max_vocab_size=config.model.vocab_size,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout
    )
    
    decoder = OptimizedUniversalDecoder(
        encoder_dim=config.model.hidden_dim,
        decoder_dim=config.model.decoder_dim,
        vocab_size=config.model.vocab_size,
        num_layers=config.model.decoder_layers,
        num_heads=config.model.decoder_heads,
        dropout=config.model.dropout,
        max_length=config.model.max_seq_length
    )
    
    # Load pretrained weights if available
    if encoder_path.exists():
        try:
            checkpoint = torch.load(encoder_path, map_location='cpu', weights_only=False)
            missing, unexpected = encoder.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if missing or unexpected:
                logger.warning(f"⚠️ Encoder state_dict partial load — missing: {missing}, unexpected: {unexpected}")
            else:
                logger.info("✅ Loaded pretrained encoder weights")
        except Exception as e:
            logger.warning(f"⚠️ Could not load encoder weights: {e}")
    
    if decoder_path.exists():
        try:
            checkpoint = torch.load(decoder_path, map_location='cpu', weights_only=False)
            missing, unexpected = decoder.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if missing or unexpected:
                logger.warning(f"⚠️ Decoder state_dict partial load — missing: {missing}, unexpected: {unexpected}")
            else:
                logger.info("✅ Loaded pretrained decoder weights")
        except Exception as e:
            logger.warning(f"⚠️ Could not load decoder weights: {e}")
    
    return encoder, decoder


def load_datasets(config: RootConfig) -> Tuple[Any, Any]:
    """Load training and validation datasets"""
    logger.info("📚 Loading datasets...")
    
    _rdm = RuntimeDirectoryManager(config=config)
    train_path = _rdm.train_final_path
    val_path = _rdm.val_final_path
    
    if not train_path.exists():
        logger.error(f"Training data not found at {train_path}")
        logger.info("Please run the data pipeline first: python -m pipeline.data.orchestrator")
        sys.exit(1)
    
    train_dataset = ModernParallelDataset(
        str(train_path),
        cache_dir=getattr(config.data, 'cache_dir', None),
        vocab_dir=_rdm.vocab_dir,
        config=config
    )
    
    val_dataset = ModernParallelDataset(
        str(val_path),
        cache_dir=getattr(config.data, 'cache_dir', None),
        vocab_dir=_rdm.vocab_dir,
        config=config
    )
    
    logger.info(f"✅ Loaded {len(train_dataset)} training samples")
    logger.info(f"✅ Loaded {len(val_dataset)} validation samples")
    
    return train_dataset, val_dataset


def _training_config_hash(config) -> str:
    """Fingerprint the config sections that affect training output."""
    return hash_config({
        'num_epochs': config.training.num_epochs,
        'batch_size': config.training.batch_size,
        'lr': config.training.lr,
        'hidden_dim': config.model.hidden_dim,
        'decoder_dim': config.model.decoder_dim,
        'num_layers': config.model.num_layers,
        'decoder_layers': config.model.decoder_layers,
        'use_lora': config.training.use_lora,
        'mixed_precision': config.training.mixed_precision,
        'experiment_name': getattr(config.training, 'experiment_name', None),
    })


def launch_training(args: argparse.Namespace):
    """Main training launcher"""
    # Setup logging
    setup_logging(
        log_dir=args.log_dir,
        log_level=args.log_level.upper()
    )
    
    logger.info("="*60)
    logger.info("🚀 INTELLIGENT TRAINING SYSTEM")
    logger.info("="*60)
    
    # Setup environment
    setup_environment()
    
    # Load configuration (dynamic or YAML)
    config = load_configuration_dynamic_or_yaml(args.config, args.dynamic)
    
    # Override config with CLI args
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.lr = args.learning_rate
    if args.num_epochs:
        config.training.num_epochs = args.num_epochs
    
    # ── Training checkpoint auto-resume ──────────────────────────────
    ckpt_mgr = PhaseCheckpoint("training")
    ckpt_mgr.load()
    cfg_hash = _training_config_hash(config)
    
    if args.force:
        ckpt_mgr.reset()
        logger.info("🔁 Force mode: training checkpoint cleared")
    elif ckpt_mgr.is_up_to_date(cfg_hash):
        logger.info(f"✅ Training already completed (config unchanged, {config.training.num_epochs} epochs).")
        logger.info("   Use --force to re-train from scratch.")
        return
    elif ckpt_mgr.completed and ckpt_mgr.config_hash != cfg_hash:
        logger.info("⚡ Training config changed — re-training")
        ckpt_mgr.reset()
    
    # Check cross-stage dependencies: data must be complete
    from utils.pipeline_checkpoint import load_pipeline_state
    data_state = load_pipeline_state().get("data", {})
    if not data_state.get("completed"):
        logger.error("Data pipeline not yet complete. Run `uts data --pipeline` first.")
        sys.exit(1)
    
    invalidate_downstream("train")
    
    # Initialize components
    encoder, decoder = initialize_models(config)
    train_dataset, val_dataset = load_datasets(config)
    
    # Setup shutdown handler
    def cleanup():
        logger.info("🛑 Graceful shutdown initiated...")
    
    shutdown_handler = GracefulShutdown(cleanup_func=cleanup)
    
    # Setup model versioning
    versioning = ModelVersion()
    
    # Determine experiment name
    experiment_name = args.experiment_name or f"intelligent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Start training with resource monitoring
    with resource_monitor.monitor("training_session"):
        start_time = time.time()
        
        if args.distributed and torch.cuda.device_count() > 1:
            logger.info(f"🚀 Launching distributed training on {torch.cuda.device_count()} GPUs")
            world_size = torch.cuda.device_count()
            mp.spawn(
                distributed_worker,
                args=(world_size, encoder, decoder, train_dataset, val_dataset, 
                     config, experiment_name, args.checkpoint, shutdown_handler),
                nprocs=world_size,
                join=True
            )
        else:
            logger.info("🚀 Launching single GPU/CPU training")
            results = train_intelligent(
                encoder=encoder,
                decoder=decoder,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                config=config,
                experiment_name=experiment_name,
                resume_from=args.checkpoint,
                shutdown_handler=shutdown_handler
            )
        
        total_time = time.time() - start_time
    
    # Post-training tasks
    logger.info("="*60)
    logger.info("✅ TRAINING COMPLETED")
    logger.info(f"⏱️ Total time: {total_time/3600:.2f} hours")
    logger.info("="*60)
    
    # Register final model
    final_model_path = Path(self.runtime_dirs.checkpoints_dir) / experiment_name / BEST_MODEL_FILENAME
    if final_model_path.exists():
        version = versioning.register_model(
            model_path=str(final_model_path),
            model_type="intelligent-universal",
            metrics={"training_time_hours": total_time/3600},
            metadata={
                "config": config.dict(),
                "experiment_name": experiment_name
            }
        )
        logger.info(f"📦 Model registered as version: {version}")
    
    # Mark training checkpoint complete
    ckpt_mgr.completed = True
    ckpt_mgr.config_hash = cfg_hash
    ckpt_mgr.save(checkpoint_path=str(final_model_path) if final_model_path.exists() else None)
    mark_stage_complete("train", cfg_hash, {"checkpoint": str(final_model_path) if final_model_path.exists() else None})
    
    # Generate resource report
    resource_summary = resource_monitor.get_summary()
    logger.info(f"📊 Resource usage summary: {resource_summary}")


def distributed_worker(rank: int, world_size: int, encoder, decoder, 
                      train_dataset, val_dataset, config, experiment_name,
                      checkpoint_path, shutdown_handler):
    """Worker function for distributed training"""
    # Set environment for this process
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    # Initialize process group
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # Run training
    train_intelligent(
        encoder=encoder,
        decoder=decoder,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        experiment_name=f"{experiment_name}_rank{rank}",
        resume_from=checkpoint_path,
        shutdown_handler=shutdown_handler
    )


def launch_evaluation(args: argparse.Namespace):
    """Launch model evaluation"""
    logger.info("🔍 Starting model evaluation...")
    
    # Load configuration
    config = load_configuration_dynamic_or_yaml(args.config, dynamic=False)
    
    # Initialize models
    encoder, decoder = initialize_models(config)
    
    # Load checkpoint
    if not args.checkpoint:
        logger.error("❌ Checkpoint path required for evaluation")
        sys.exit(1)
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    # Load test dataset
    test_dataset = ModernParallelDataset(
        args.test_data or str(RuntimeDirectoryManager(config=config).processed_dir / TEST_FINAL_FILENAME),
        vocab_dir=RuntimeDirectoryManager(config=config).vocab_dir
    )
    
    # Run evaluation
    from evaluation.evaluator import evaluate_translation_quality
    
    results = evaluate_translation_quality(
        encoder_model=encoder,
        decoder_model=decoder,
        vocab_manager=test_dataset.vocab_manager if hasattr(test_dataset, 'vocab_manager') else None,
        test_data_path=args.test_data or str(RuntimeDirectoryManager(config=config).processed_dir / TEST_FINAL_FILENAME),
    )
    
    logger.info(f"📊 Evaluation results: {results}")
    
    # Save results
    results_path = Path(args.output_dir) / EVALUATION_REPORT_FILENAME
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✅ Results saved to {results_path}")


def launch_profiling(args: argparse.Namespace):
    """Launch training profiling"""
    logger.info("🔬 Starting training profiling...")
    
    # Load configuration
    config = load_configuration_dynamic_or_yaml(args.config, dynamic=False)
    
    # Initialize components
    encoder, decoder = initialize_models(config)
    train_dataset, _ = load_datasets(config)
    
    # Create trainer
    trainer = IntelligentTrainer(
        encoder=encoder,
        decoder=decoder,
        train_dataset=train_dataset,
        val_dataset=train_dataset,  # Use train for profiling
        config=config,
        experiment_name="profiling"
    )
    
    # Run profiling
    from pipeline.training.profiling import ProfileGuidedTrainer
    
    profiler = ProfileGuidedTrainer(trainer)
    results = profiler.profile_training_step(
        num_steps=args.profile_steps or 10,
        trace_path=args.output_dir
    )
    
    # Benchmark configurations if requested
    if args.benchmark:
        configs = [
            {'name': 'baseline', 'batch_size': 32, 'mixed_precision': False},
            {'name': 'mixed_precision', 'batch_size': 32, 'mixed_precision': True},
            {'name': 'large_batch', 'batch_size': 64, 'mixed_precision': True},
            {'name': 'compiled', 'batch_size': 32, 'compile_model': True},
        ]
        
        benchmark_results = profiler.benchmark_configurations(configs)
        logger.info(f"📊 Benchmark results:\n{benchmark_results}")


def main():
    """Main entry point with CLI"""
    parser = argparse.ArgumentParser(
        description="Intelligent Universal Translation System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--config', type=str, required=True,
                            help="Path to configuration file or 'dynamic'")
    train_parser.add_argument('--dynamic', action='store_true',
                            help='Use dynamic config generation (no YAML)')
    train_parser.add_argument('--experiment-name', type=str,
                            help='Name for this experiment')
    train_parser.add_argument('--checkpoint', type=str,
                            help='Resume from checkpoint')
    train_parser.add_argument('--distributed', action='store_true',
                            help='Use distributed training if available')
    train_parser.add_argument('--batch-size', type=int,
                            help='Override batch size')
    train_parser.add_argument('--learning-rate', type=float,
                            help='Override learning rate')
    train_parser.add_argument('--num-epochs', type=int,
                            help='Override number of epochs')
    train_parser.add_argument('--force', action='store_true',
                            help='Ignore training checkpoint and re-train from scratch')
    train_parser.add_argument('--log-dir', type=str, default=str(RuntimeDirectoryManager().logs_dir),
                            help='Directory for logs')
    train_parser.add_argument('--log-level', type=str, default='info',
                            choices=['debug', 'info', 'warning', 'error'],
                            help='Logging level')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a model')
    eval_parser.add_argument('--config', type=str, required=True,
                           help='Path to configuration file')
    eval_parser.add_argument('--checkpoint', type=str, required=True,
                           help='Path to model checkpoint')
    eval_parser.add_argument('--test-data', type=str,
                           help='Path to test data')
    eval_parser.add_argument('--batch-size', type=int,
                           help='Batch size for evaluation')
    eval_parser.add_argument('--output-dir', type=str, default='results',
                           help='Output directory for results')
    
    # Profiling command
    profile_parser = subparsers.add_parser('profile', help='Profile training')
    profile_parser.add_argument('--config', type=str, required=True,
                              help='Path to configuration file')
    profile_parser.add_argument('--profile-steps', type=int, default=10,
                              help='Number of steps to profile')
    profile_parser.add_argument('--benchmark', action='store_true',
                              help='Run configuration benchmarks')
    profile_parser.add_argument('--output-dir', type=str, default='profiling',
                              help='Output directory for profiling results')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare experiments')
    compare_parser.add_argument('--experiments', nargs='+', required=True,
                              help='List of experiment directories')
    compare_parser.add_argument('--output-dir', type=str, default='comparisons',
                              help='Output directory for comparisons')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Visualize training')
    viz_parser.add_argument('--experiment-dir', type=str, required=True,
                          help='Experiment directory')
    viz_parser.add_argument('--output-dir', type=str, default='visualizations',
                          help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    if args.command == 'train':
        launch_training(args)
    elif args.command == 'evaluate':
        launch_evaluation(args)
    elif args.command == 'profile':
        launch_profiling(args)
    elif args.command == 'compare':
        from tools.compare import ExperimentComparator
        comparator = ExperimentComparator(args.experiments, args.output_dir)
        comparator.generate_comparison_report()
        comparator.plot_learning_curves()
        comparator.plot_metrics_comparison()
    elif args.command == 'visualize':
        from tools.visualize import TrainingDashboard
        # Would need to load trainer or metrics here
        logger.info("Visualization command needs implementation based on your setup")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Run main
    main()