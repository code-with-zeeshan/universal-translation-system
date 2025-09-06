# training/launch.py
"""
Launch utilities for intelligent training system
Handles CLI, distributed setup, and orchestration
"""

import argparse
import sys
import os
import torch
import torch.multiprocessing as mp
from pathlib import Path
import logging
import json
import yaml
from typing import Optional, Dict, Any, Tuple
import time
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from training.intelligent_trainer import IntelligentTrainer, train_intelligent
from encoder.universal_encoder import UniversalEncoder
from cloud_decoder.optimized_decoder import OptimizedUniversalDecoder
from data.dataset_classes import ModernParallelDataset
from utils.shutdown_handler import GracefulShutdown
from utils.model_versioning import ModelVersion
from utils.resource_monitor import resource_monitor
from utils.logging_config import setup_logging
from config.schemas import RootConfig, load_config as load_pydantic_config

# Centralized logging for training
setup_logging(log_dir="logs", log_level=os.environ.get("LOG_LEVEL", "INFO"))
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
        from config.schemas import RootConfig, DataConfig, ModelConfig, TrainingConfig, MemoryConfig, VocabularyConfig
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
    """Initialize encoder and decoder models"""
    logger.info("🔧 Initializing models...")
    
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
        dropout=config.model.dropout
    )
    
    # Load pretrained weights if available
    encoder_path = Path('models/encoder/universal_encoder_initial.pt')
    decoder_path = Path('models/decoder/universal_decoder_initial.pt')
    
    if encoder_path.exists():
        try:
            checkpoint = torch.load(encoder_path, map_location='cpu')
            encoder.load_state_dict(checkpoint['model_state_dict'])
            logger.info("✅ Loaded pretrained encoder weights")
        except Exception as e:
            logger.warning(f"⚠️ Could not load encoder weights: {e}")
    
    if decoder_path.exists():
        try:
            checkpoint = torch.load(decoder_path, map_location='cpu')
            decoder.load_state_dict(checkpoint['model_state_dict'])
            logger.info("✅ Loaded pretrained decoder weights")
        except Exception as e:
            logger.warning(f"⚠️ Could not load decoder weights: {e}")
    
    return encoder, decoder


def load_datasets(config: RootConfig) -> Tuple[Any, Any]:
    """Load training and validation datasets"""
    logger.info("📚 Loading datasets...")
    
    train_path = Path(config.data.processed_dir) / 'train_final.txt'
    val_path = Path(config.data.processed_dir) / 'val_final.txt'
    
    if not train_path.exists():
        logger.error(f"Training data not found at {train_path}")
        logger.info("Please run the data pipeline first: python -m data.unified_data_pipeline")
        sys.exit(1)
    
    train_dataset = ModernParallelDataset(
        str(train_path),
        cache_dir=getattr(config.data, 'cache_dir', None),
        vocab_dir=config.vocabulary.vocab_dir,
        config=config
    )
    
    val_dataset = ModernParallelDataset(
        str(val_path),
        cache_dir=getattr(config.data, 'cache_dir', None),
        vocab_dir=config.vocabulary.vocab_dir,
        config=config
    )
    
    logger.info(f"✅ Loaded {len(train_dataset)} training samples")
    logger.info(f"✅ Loaded {len(val_dataset)} validation samples")
    
    return train_dataset, val_dataset


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
        config.training.learning_rate = args.learning_rate
    if args.num_epochs:
        config.training.num_epochs = args.num_epochs
    
    # Initialize components
    encoder, decoder = initialize_models(config)
    train_dataset, val_dataset = load_datasets(config)
    
    # Setup shutdown handler
    def cleanup():
        logger.info("🛑 Graceful shutdown initiated...")
        # Save emergency checkpoint handled by trainer
    
    shutdown_handler = GracefulShutdown(cleanup_func=cleanup)
    
    # Setup model versioning (default directory: "models")
    versioning = ModelVersion()
    
    # Determine experiment name
    experiment_name = args.experiment_name or f"intelligent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Start training with resource monitoring
    with resource_monitor.monitor("training_session"):
        start_time = time.time()
        
        # Launch training
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
            
            # Direct training
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
    final_model_path = Path(config.training.checkpoint_dir) / experiment_name / "best_model.pt"
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
    config = load_configuration(args.config)
    
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
        args.test_data or str(Path(config.data.processed_dir) / 'test_final.txt'),
        vocab_dir=config.vocabulary.vocab_dir
    )
    
    # Run evaluation
    from evaluation.evaluate_model import evaluate_model
    
    results = evaluate_model(
        encoder=encoder,
        decoder=decoder,
        test_dataset=test_dataset,
        config=config,
        batch_size=args.batch_size or config.training.batch_size
    )
    
    logger.info(f"📊 Evaluation results: {results}")
    
    # Save results
    results_path = Path(args.output_dir) / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✅ Results saved to {results_path}")


def launch_profiling(args: argparse.Namespace):
    """Launch training profiling"""
    logger.info("🔬 Starting training profiling...")
    
    # Load configuration
    config = load_configuration(args.config)
    
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
    from training.profiling import ProfileGuidedTrainer
    
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
    train_parser.add_argument('--log-dir', type=str, default='logs',
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
        from training.comparison import ExperimentComparator
        comparator = ExperimentComparator(args.experiments, args.output_dir)
        comparator.generate_comparison_report()
        comparator.plot_learning_curves()
        comparator.plot_metrics_comparison()
    elif args.command == 'visualize':
        from training.visualization import TrainingDashboard
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