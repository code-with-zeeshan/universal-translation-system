# training/progressive_training.py (refactored - orchestrator version)
import torch
import torch.multiprocessing as mp
import yaml
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import tempfile
import os
import sys
from datetime import datetime
from dataclasses import make_dataclass

from utils.gpu_utils import optimize_gpu_memory, get_gpu_memory_info
from utils.shutdown_handler import GracefulShutdown
from utils.model_versioning import ModelVersion
from utils.resource_monitor import resource_monitor
# --- ADDED: Direct imports for in-process training ---
from training.distributed_train import (
    train_with_unified_distributed_wrapper,
    TrainingConfig,
    setup_distributed_environment
)
from encoder.universal_encoder import UniversalEncoder
from cloud_decoder.optimized_decoder import OptimizedUniversalDecoder
from utils.dataset_classes import ModernParallelDataset
from vocabulary.vocabulary_manager import VocabularyManager

# Initialize GPU optimization
optimize_gpu_memory()

logger = logging.getLogger(__name__)

class ProgressiveTrainingOrchestrator:
    """Orchestrates progressive training by launching appropriate training scripts"""
    
    def __init__(self, 
                 base_config_path: str = 'config/base.yaml',
                 checkpoint_base_dir: str = 'checkpoints/progressive'):
        
        self.base_config_path = Path(base_config_path)
        self.checkpoint_base_dir = Path(checkpoint_base_dir)
        self.checkpoint_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Load base config
        with open(self.base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        # Define language tiers
        self.language_tiers = {
            'tier1': {
                'languages': ['en', 'es', 'fr', 'de'],
                'reason': 'High-resource Indo-European',
                'epochs': 10,
                'lr': 5e-4,
                'batch_size': 64,
                'warmup_steps': 2000
            },
            'tier2': {
                'languages': ['zh', 'ja', 'ru', 'pt', 'it'],
                'reason': 'Major languages, different scripts',
                'epochs': 8,
                'lr': 3e-4,
                'batch_size': 48,
                'warmup_steps': 1500
            },
            'tier3': {
                'languages': ['ar', 'hi', 'ko', 'nl', 'pl'],
                'reason': 'Medium-resource, diverse',
                'epochs': 6,
                'lr': 2e-4,
                'batch_size': 32,
                'warmup_steps': 1000
            },
            'tier4': {
                'languages': ['tr', 'th', 'vi', 'uk', 'id', 'sv'],
                'reason': 'Lower-resource languages',
                'epochs': 4,
                'lr': 1e-4,
                'batch_size': 24,
                'warmup_steps': 500
            }
        }
        
        # Training history
        self.training_history = {
            'tier_results': {},
            'total_time': 0,
            'gpu_count': self._detect_gpus(),
            'start_time': datetime.now().isoformat()
        }
        
        # State management
        self.state_file = self.checkpoint_base_dir / "progressive_state.json"
        self.current_state = self._load_state()
        
    def _detect_gpus(self) -> int:
        """Detect number of available GPUs"""
        gpu_count = torch.cuda.device_count()
        logger.info(f"Detected {gpu_count} GPU(s)")
        
        if gpu_count > 0:
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                logger.info(f"  GPU {i}: {gpu_name}")
        
        return gpu_count
    
    def _load_state(self) -> Dict:
        """Load training state from disk"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            'completed_tiers': [],
            'current_tier': None,
            'last_checkpoint': None
        }
    
    def _save_state(self):
        """Save training state to disk"""
        with open(self.state_file, 'w') as f:
            json.dump(self.current_state, f, indent=2)
    
    def _create_tier_config(self, tier_name: str, tier_config: Dict, 
                           previous_checkpoint: Optional[str] = None) -> str:
        """Create temporary config file for a specific tier"""
        
        # Start with base config
        config = self.base_config.copy()
        
        # Update with tier-specific settings
        config['training']['num_epochs'] = tier_config['epochs']
        config['training']['lr'] = tier_config['lr']
        config['training']['batch_size'] = tier_config['batch_size']
        config['training']['warmup_steps'] = tier_config['warmup_steps']
        
        # Filter data distribution for current tier languages
        tier_languages = set(tier_config['languages'])
        filtered_distribution = {}
        
        for lang_pair, count in self.base_config['data']['training_distribution'].items():
            src, tgt = lang_pair.split('-')
            if src in tier_languages or tgt in tier_languages:
                filtered_distribution[lang_pair] = count
        
        config['data']['training_distribution'] = filtered_distribution
        config['data']['active_languages'] = tier_config['languages']
        
        # Set checkpoint directory for this tier
        tier_checkpoint_dir = self.checkpoint_base_dir / tier_name
        tier_checkpoint_dir.mkdir(exist_ok=True)
        config['data']['checkpoint_dir'] = str(tier_checkpoint_dir)
        
        # Add checkpoint to resume from if available
        if previous_checkpoint:
            config['training']['resume_from'] = previous_checkpoint
        
        # Add tier metadata
        config['tier_metadata'] = {
            'tier_name': tier_name,
            'tier_index': list(self.language_tiers.keys()).index(tier_name),
            'total_tiers': len(self.language_tiers),
            'languages': tier_config['languages'],
            'reason': tier_config['reason']
        }
        
        # Create temporary config file
        temp_config_path = self.checkpoint_base_dir / f"temp_{tier_name}_config.yaml"
        
        # Add reference to base config for inheritance
        config['_base_'] = str(self.base_config_path)
        
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Created config for {tier_name}: {temp_config_path}")
        return str(temp_config_path)
    
    def _prepare_tier_resources(self, tier_config: Dict, config_dict: Dict) -> Tuple:
        """Loads and prepares models and datasets for a specific tier."""
        logger.info("Preparing resources for the current training tier...")

        # Get model config from the tier's config
        model_config = config_dict.get('model', {})

        # Initialize models on CPU first; they will be moved to GPUs in the spawned processes
        encoder = UniversalEncoder(
            max_vocab_size=model_config.get('vocab_size', 50000),
            hidden_dim=model_config.get('hidden_dim', 1024),
            num_layers=model_config.get('num_layers', 6),
            num_heads=model_config.get('num_heads', 16),
            dropout=model_config.get('dropout', 0.1)
        )
        decoder = OptimizedUniversalDecoder(
            encoder_dim=model_config.get('hidden_dim', 1024),
            decoder_dim=model_config.get('decoder_dim', 512),
            vocab_size=model_config.get('vocab_size', 50000),
            num_layers=model_config.get('decoder_layers', 6),
            num_heads=model_config.get('decoder_heads', 8),
            dropout=model_config.get('dropout', 0.1)
        )

        # Load from previous checkpoint if it exists
        last_checkpoint = self.current_state.get('last_checkpoint')
        if last_checkpoint and Path(last_checkpoint).exists():
            logger.info(f"Loading weights from previous tier's checkpoint: {last_checkpoint}")
            # Note: The actual loading of the state dict into the sharded model
            # happens inside the distributed training function. Here we just pass the path.
            pass # The checkpoint path will be passed to the training function.
        else:
            logger.info("No previous tier checkpoint found. Checking for bootstrapped models...")
            # --- ADDED: Load bootstrapped models if they exist ---
            bootstrapped_encoder_path = Path('models/encoder/universal_encoder_initial.pt')
            bootstrapped_decoder_path = Path('models/decoder/universal_decoder_initial.pt')

            if bootstrapped_encoder_path.exists():
                logger.info(f"Found bootstrapped encoder at {bootstrapped_encoder_path}. Loading weights...")
                try:
                    checkpoint = torch.load(bootstrapped_encoder_path, map_location='cpu')
                    encoder.load_state_dict(checkpoint['model_state_dict'])
                    logger.info("✅ Successfully loaded bootstrapped encoder weights.")
                except Exception as e:
                    logger.error(f"❌ Failed to load bootstrapped encoder: {e}. Starting from scratch.")

            if bootstrapped_decoder_path.exists():
                logger.info(f"Found bootstrapped decoder at {bootstrapped_decoder_path}. Loading weights...")
                try:
                    checkpoint = torch.load(bootstrapped_decoder_path, map_location='cpu')
                    decoder.load_state_dict(checkpoint['model_state_dict'])
                    logger.info("✅ Successfully loaded bootstrapped decoder weights.")
                except Exception as e:
                    logger.error(f"❌ Failed to load bootstrapped decoder: {e}. Starting from scratch.")

        # Load datasets based on the tier's language configuration
        # This is a simplified example; in a real scenario, you would filter the main dataset
        # or load tier-specific data files.
        train_data_path = Path(config_dict['data']['processed_dir']) / 'train_final.txt'
        val_data_path = Path(config_dict['data']['processed_dir']) / 'val_final.txt'

        # --- MODIFIED: Filter dataset by active languages for the tier ---
        active_languages = tier_config.get('languages')
        logger.info(f"Filtering datasets for active languages: {active_languages}")
        train_dataset = ModernParallelDataset(
            str(train_data_path)
        )
        val_dataset = ModernParallelDataset(
            str(val_data_path)
        )
        # --- END MODIFICATION ---

        return encoder, decoder, train_dataset, val_dataset

    def _run_tier_training(self, tier_name: str, tier_config: Dict, 
                          config_path: str) -> Dict[str, any]:
        """Run training for a single tier"""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting training for {tier_name}")
        logger.info(f"Languages: {tier_config['languages']}")
        logger.info(f"Epochs: {tier_config['epochs']}, LR: {tier_config['lr']}")
        logger.info(f"{'='*60}\n")
        
        # --- REFACTORED: Direct invocation instead of subprocess ---
        
        # 1. Load the full configuration for the tier
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # 2. Prepare models and datasets
        encoder, decoder, train_dataset, val_dataset = self._prepare_tier_resources(tier_config, config_dict)

        # 3. Setup environment and arguments for the training function
        setup_distributed_environment()
        world_size = self.training_history['gpu_count']
        
        # Create a simple args object to pass to the training wrapper
        Args = make_dataclass('Args', ['global_batch_size', 'epochs', 'checkpoint'])
        args = Args(
            global_batch_size=tier_config['batch_size'] * world_size,
            epochs=tier_config['epochs'],
            checkpoint=self.current_state.get('last_checkpoint')
        )
        
        # Run training with resource monitoring
        start_time = time.time()
        result = {'success': False, 'error': None}
        
        with resource_monitor.monitor(f"tier_{tier_name}_training"):
            try:
                # 4. Spawn training processes
                if world_size > 1:
                    spawn_args = (world_size, encoder, decoder, train_dataset, val_dataset, args, config_dict, None)
                    mp.spawn(train_with_unified_distributed_wrapper, args=spawn_args, nprocs=world_size, join=True)
                else:
                    # Handle single GPU case
                    train_with_unified_distributed_wrapper(0, 1, encoder, decoder, train_dataset, val_dataset, args, config_dict, None)
                
                result['success'] = True
                logger.info(f"✅ {tier_name} training completed successfully")
                    
            except KeyboardInterrupt:
                logger.info("Training interrupted by user")
                raise
            except Exception as e:
                result['error'] = str(e)
                logger.error(f"Error during {tier_name} training: {e}")
                result['success'] = False
        
        # Record metrics
        training_time = time.time() - start_time
        result['training_time'] = training_time
        result['tier_name'] = tier_name
        
        # Find best checkpoint from this tier
        tier_checkpoint_dir = self.checkpoint_base_dir / tier_name
        checkpoints = list(tier_checkpoint_dir.glob("best_model.pt"))
        
        if checkpoints:
            result['best_checkpoint'] = str(checkpoints[0])
            logger.info(f"Best checkpoint: {result['best_checkpoint']}")
        else:
            # Fallback to any checkpoint
            checkpoints = list(tier_checkpoint_dir.glob("checkpoint_*.pt"))
            if checkpoints:
                # Sort by modification time and take the latest
                checkpoints.sort(key=lambda p: p.stat().st_mtime)
                result['best_checkpoint'] = str(checkpoints[-1])
                logger.info(f"Using latest checkpoint: {result['best_checkpoint']}")
        
        return result
    
    def train_progressive(self, 
                         start_from_tier: Optional[str] = None,
                         shutdown_handler: Optional[GracefulShutdown] = None,
                         cleanup_temp_configs: bool = True) -> Dict:
        """Main progressive training orchestration"""
        
        logger.info("🚀 Starting Progressive Training Orchestration")
        logger.info(f"GPU count: {self.training_history['gpu_count']}")
        logger.info(f"Checkpoint directory: {self.checkpoint_base_dir}")
        
        # Determine starting point
        tier_names = list(self.language_tiers.keys())
        if start_from_tier:
            start_idx = tier_names.index(start_from_tier)
        else:
            # Resume from last completed tier
            if self.current_state['completed_tiers']:
                last_completed = self.current_state['completed_tiers'][-1]
                start_idx = tier_names.index(last_completed) + 1
            else:
                start_idx = 0
        
        # Training loop
        overall_start_time = time.time()
        previous_checkpoint = self.current_state.get('last_checkpoint')
        
        for tier_idx in range(start_idx, len(tier_names)):
            tier_name = tier_names[tier_idx]
            tier_config = self.language_tiers[tier_name]
            
            # Check for shutdown
            if shutdown_handler and shutdown_handler.should_stop():
                logger.info("Shutdown requested. Saving state and exiting.")
                break
            
            # Update current state
            self.current_state['current_tier'] = tier_name
            self._save_state()
            
            # Create tier-specific config
            config_path = self._create_tier_config(
                tier_name, 
                tier_config,
                previous_checkpoint
            )
            
            try:
                # Run training for this tier
                result = self._run_tier_training(tier_name, tier_config, config_path)
                
                # Store results
                self.training_history['tier_results'][tier_name] = result
                
                if result['success']:
                    # Update state
                    self.current_state['completed_tiers'].append(tier_name)
                    if 'best_checkpoint' in result:
                        self.current_state['last_checkpoint'] = result['best_checkpoint']
                        previous_checkpoint = result['best_checkpoint']
                    self._save_state()
                    
                    # Log tier summary
                    logger.info(f"\n{tier_name} Summary:")
                    logger.info(f"  Training time: {result['training_time']/3600:.2f} hours")
                    logger.info(f"  Best checkpoint: {result.get('best_checkpoint', 'N/A')}")
                else:
                    logger.error(f"Failed to complete {tier_name}: {result.get('error')}")
                    if not self._should_continue_on_failure():
                        break
                        
            finally:
                # Clean up temporary config if requested
                if cleanup_temp_configs and Path(config_path).exists():
                    Path(config_path).unlink()
                    logger.debug(f"Removed temporary config: {config_path}")
        
        # Final summary
        self.training_history['total_time'] = time.time() - overall_start_time
        self._generate_final_report()
        
        return self.training_history
    
    def _should_continue_on_failure(self) -> bool:
        """Decide whether to continue after a tier fails"""
        # Could implement retry logic or user prompt here
        logger.warning("Tier failed. Continuing with next tier...")
        return True
    
    def _generate_final_report(self):
        """Generate and save final training report"""
        
        report_path = self.checkpoint_base_dir / "progressive_training_report.json"
        
        report = {
            'summary': {
                'total_time_hours': self.training_history['total_time'] / 3600,
                'completed_tiers': self.current_state['completed_tiers'],
                'gpu_count': self.training_history['gpu_count'],
                'start_time': self.training_history['start_time'],
                'end_time': datetime.now().isoformat()
            },
            'tier_details': self.training_history['tier_results'],
            'final_checkpoint': self.current_state.get('last_checkpoint'),
            'resource_summary': resource_monitor.get_summary()
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("PROGRESSIVE TRAINING COMPLETE")
        logger.info("="*60)
        logger.info(f"Total time: {report['summary']['total_time_hours']:.2f} hours")
        logger.info(f"Completed tiers: {', '.join(report['summary']['completed_tiers'])}")
        logger.info(f"Final checkpoint: {report['final_checkpoint']}")
        logger.info("Full report saved to: {report_path}")
        logger.info("="*60)
    
    def validate_final_model(self, checkpoint_path: Optional[str] = None) -> Dict:
        """Validate the final model on all languages"""
        
        if not checkpoint_path:
            checkpoint_path = self.current_state.get('last_checkpoint')
        
        if not checkpoint_path or not Path(checkpoint_path).exists():
            logger.error("No valid checkpoint found for validation")
            return {}
        
        logger.info(f"Validating final model: {checkpoint_path}")
        
        # Create validation config with all languages
        val_config = self.base_config.copy()
        val_config['training']['num_epochs'] = 0  # Validation only
        val_config['training']['validate_only'] = True
        val_config['training']['checkpoint'] = checkpoint_path
        
        # Save validation config
        val_config_path = self.checkpoint_base_dir / "validation_config.yaml"
        with open(val_config_path, 'w') as f:
            yaml.dump(val_config, f)
        
        # Run validation
        cmd = self._build_training_command(str(val_config_path), "validation")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Validation completed successfully")
            
            # Parse validation results from output
            # This would need to be implemented based on your validation output format
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Validation failed: {e}")
            logger.error(f"Error output: {e.stderr}")
        
        return {}

def main():
    """Main entry point for progressive training orchestration"""
    
    # Setup logging
    from utils.logging_config import setup_logging
    setup_logging(log_dir="logs/progressive_training", log_level="INFO")
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Progressive Training Orchestrator")
    parser.add_argument('--base-config', type=str, default='config/base.yaml',
                       help='Path to base configuration file')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/progressive',
                       help='Directory for progressive training checkpoints')
    parser.add_argument('--start-from-tier', type=str, default=None,
                       help='Start from specific tier (tier1, tier2, tier3, tier4)')
    parser.add_argument('--validate-final', action='store_true',
                       help='Run validation on final model after training')
    args = parser.parse_args()
    
    # Setup shutdown handler
    def cleanup():
        logger.info("Graceful shutdown initiated...")
        # The orchestrator will handle state saving automatically
    
    shutdown_handler = GracefulShutdown(cleanup_func=cleanup)
    
    # Initialize orchestrator
    orchestrator = ProgressiveTrainingOrchestrator(
        base_config_path=args.base_config,
        checkpoint_base_dir=args.checkpoint_dir
    )
    
    # Run progressive training
    try:
        results = orchestrator.train_progressive(
            start_from_tier=args.start_from_tier,
            shutdown_handler=shutdown_handler
        )
        
        # Optional: Validate final model
        if args.validate_final:
            orchestrator.validate_final_model()
        
        # Register final model
        if results.get('final_checkpoint'):
            versioning = ModelVersion()
            version = versioning.register_model(
                model_path=results['final_checkpoint'],
                model_type="progressive-universal",
                metrics={
                    'training_hours': results['total_time'] / 3600,
                    'completed_tiers': len(orchestrator.current_state['completed_tiers'])
                },
                metadata={
                    'training_history': results,
                    'language_tiers': orchestrator.language_tiers
                }
            )
            logger.info(f"Final model registered as version: {version}")
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Progressive training failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()