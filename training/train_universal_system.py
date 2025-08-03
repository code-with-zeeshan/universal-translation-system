# training/train_universal_system.py (main)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import wandb
from tqdm import tqdm
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json
import time
from contextlib import contextmanager
from data.custom_samplers import TemperatureSampler
from vocabulary.vocabulary_manager import VocabularyManager
import yaml
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
import torch.utils.benchmark as benchmark
import pandas as pd
import numpy as np
from collections import defaultdict
import os
import subprocess
from utils.gpu_utils import optimize_gpu_memory, get_gpu_memory_info
from utils.resource_monitor import resource_monitor
from utils.shutdown_handler import GracefulShutdown
from utils.model_versioning import ModelVersion

# Initialize GPU optimization
optimize_gpu_memory()

# Define checkpoint version constant
CHECKPOINT_VERSION = "2.0"

try:
    import safetensors.torch
except ImportError:
    logger.warning("safetensors not available, using standard torch.save")
    safetensors = None

 # Expanded GPU configuration map including new and common hardware.
 # The order matters: more specific names (e.g., "RTX 3090") should come before general ones ("RTX").

GPU_CONFIG_MAP = [
    # Datacenter GPUs (NVIDIA)
    ("H100", "config/training_h100.yaml"),
    ("A100", "config/training_a100.yaml"),
    ("V100", "config/training_v100.yaml"),
    ("L4", "config/training_l4.yaml"),
    ("T4", "config/training_t4.yaml"),
    
    # Datacenter GPUs (AMD)
    ("MI250", "config/training_amd_mi250.yaml"),
    ("MI300", "config/training_amd_mi250.yaml"), # MI300 can use the same config as MI250
    
    # High-End Consumer GPUs (NVIDIA)
    ("RTX 4090", "config/training_rtx4090.yaml"),
    ("RTX 3090", "config/training_rtx3090.yaml"),
    ("RTX 3080", "config/training_rtx3080.yaml"),
    
    # Mid-Range Consumer GPUs (NVIDIA)
    ("RTX 3060", "config/training_rtx3060.yaml"),
    
    # Google Colab Free Tier (often older GPUs)
    ("K80", "config/training_colab_free.yaml"),
]

def auto_select_config(
    languages: Optional[List[str]] = None,
    data_info: Optional[Dict[str, int]] = None
) -> str:
    """
    Automatically selects the best training configuration file based on the detected
    hardware, language count, and data size.
    Handles NVIDIA CUDA, AMD ROCm, and CPU-only environments.
    """
    # Check for AMD ROCm environment first
    try:
        # The 'rocm-smi' command is the equivalent of 'nvidia-smi' for AMD GPUs
        result = subprocess.run(['rocm-smi', '--showproductname'], capture_output=True, text=True, check=True)
        gpu_name = result.stdout.strip()
        print(f"Detected AMD ROCm GPU: {gpu_name}")
        for key, config_path in GPU_CONFIG_MAP:
            if key in gpu_name:
                print(f"Using config: {config_path}")
                return config_path
    except (FileNotFoundError, subprocess.CalledProcessError):
        # rocm-smi not found or failed, proceed to check for NVIDIA CUDA
        pass

    # Check for NVIDIA CUDA environment
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Detected NVIDIA CUDA GPU: {gpu_name}")
    else:
        # CPU-only configuration
        print("No compatible GPU detected. Using CPU-only configuration.")
        return "config/training_cpu.yaml"    

    # Smart logic based on task complexity 
    num_languages = len(languages) if languages else 0
    total_sentences = sum(data_info.values()) if data_info else 0
    
    # Rule 1: Large-scale multilingual training on high-end GPUs
    if num_languages > 20 and total_sentences > 50_000_000:
        if "H100" in gpu_name or "A100" in gpu_name:
            print("INFO: High language count and data size on A100/H100. Selecting FSDP-optimized config.")
            return "config/training_a100_fsdp.yaml" # Assumes you create this config
            
    # Rule 2: Fine-tuning a few languages
    if 1 < num_languages <= 5:
        if "V100" in gpu_name or "RTX 3090" in gpu_name or "RTX 4090" in gpu_name:
            print("INFO: Small language set. Selecting fine-tuning optimized config.")
            return "config/training_v100_finetune.yaml" # Assumes you create this config

    for key, config_path in GPU_CONFIG_MAP:
        if key in gpu_name:
            print(f"Using default config for {key}: {config_path}")
            return config_path
        
    # Fallback for unknown NVIDIA GPUs
    print(f"Unknown NVIDIA GPU: {gpu_name}. Using default T4 config as a safe fallback.")
    return "config/training_t4.yaml"

def load_config(config_path: str) -> dict:
    """
    Loads and merges hierarchical YAML configurations.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    # Handle inheritance from a _base_ file
    if '_base_' in config:
        base_path = path.parent / config['_base_']
        base_config = load_config(str(base_path))
        
        # Deep merge the configurations
        import collections.abc
        def deep_merge(d, u):
            for k, v in u.items():
                if isinstance(v, collections.abc.Mapping):
                    d[k] = deep_merge(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
            
        config = deep_merge(base_config, config)
        del config['_base_']

    return config

from training.memory_efficient_training import (
    MemoryOptimizedTrainer, 
    MemoryConfig, 
    DynamicBatchSizer as BaseDynamicBatchSizer,
    create_modern_training_setup
)

logger = logging.getLogger(__name__)

class ModernUniversalSystemTrainer:
    """Modern universal system trainer with all latest optimizations"""
    
    def __init__(self, encoder, decoder, train_data_path, val_data_path, 
                 config: MemoryConfig = None, experiment_name: str = "universal-translation"):
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.experiment_name = experiment_name
        
        # Initialize memory optimized trainer
        self.memory_config = config or MemoryConfig()
        self.trainer = MemoryOptimizedTrainer(encoder, self.memory_config)
        
        # Move models to device
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        
        # Setup optimized models
        self._setup_models()
        
        # Create datasets
        self.train_dataset = ModernParallelDataset(train_data_path)
        self.val_dataset = ModernParallelDataset(val_data_path)
        
        # Create modern optimizers
        self.encoder_optimizer = self.trainer.create_optimized_optimizer(
            self.encoder.parameters(), lr=5e-5, weight_decay=0.01
        )
        self.decoder_optimizer = self.trainer.create_optimized_optimizer(
            self.decoder.parameters(), lr=5e-5, weight_decay=0.01
        )
        
        # Dynamic batch sizing
        self.batch_sizer = DynamicBatchSizer(initial_batch_size=32, max_batch_size=128)
        
        # Initialize wandb with modern config
        self._setup_wandb()
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.memory_config.mixed_precision else None
        
        logger.info("🚀 Modern Universal System Trainer initialized!")

    # Add to ModernUniversalSystemTrainer.__init__
    def _setup_reproducibility(self, seed: int = 42):
        """Setup training reproducibility"""
        import random
        import numpy as np
    
        # Set all random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
        # Set deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
        # For newer PyTorch versions
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
    
        logger.info(f"🎲 Reproducibility setup with seed: {seed}")    
    
    def _setup_models(self):
        """Setup models with modern optimizations"""
        
        # Apply memory optimizations to both models
        if self.memory_config.compile_model:
            try:
                self.encoder = torch.compile(
                    self.encoder, 
                    mode=self.memory_config.compile_mode,
                    dynamic=True
                )
                self.decoder = torch.compile(
                    self.decoder, 
                    mode=self.memory_config.compile_mode,
                    dynamic=True
                )
                logger.info("✅ Models compiled successfully")
            except Exception as e:
                logger.warning(f"⚠️ Model compilation failed: {e}")
        
        # Enable channels last if configured
        if self.memory_config.use_channels_last:
            try:
                self.encoder = self.encoder.to(memory_format=torch.channels_last)
                self.decoder = self.decoder.to(memory_format=torch.channels_last)
                logger.info("✅ Channels last memory format enabled")
            except Exception as e:
                logger.warning(f"⚠️ Channels last format failed: {e}")
    
    def _setup_wandb(self):
        """Setup wandb with comprehensive config"""
        
        wandb_config = {
            "model": {
                "encoder_params": sum(p.numel() for p in self.encoder.parameters()),
                "decoder_params": sum(p.numel() for p in self.decoder.parameters()),
                "total_params": sum(p.numel() for p in self.encoder.parameters()) + 
                              sum(p.numel() for p in self.decoder.parameters())
            },
            "training": {
                "mixed_precision": self.memory_config.mixed_precision,
                "gradient_checkpointing": self.memory_config.gradient_checkpointing,
                "compile_model": self.memory_config.compile_model,
                "flash_attention": self.memory_config.use_flash_attention,
                "dtype": str(self.memory_config.dtype),
                "compile_mode": self.memory_config.compile_mode
            },
            "system": {
                "pytorch_version": torch.__version__,
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "device": str(self.device),
                "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None
            }
        }
        
        wandb.init(
            project="universal-translation", 
            name=f"{self.experiment_name}-{time.strftime('%Y%m%d-%H%M%S')}",
            config=wandb_config,
            tags=["modern", "memory-optimized", "universal-system"]
        )
    
    @contextmanager
    def training_mode(self):
        """Context manager for training mode"""
        self.encoder.train()
        self.decoder.train()
        try:
            yield
        finally:
            pass
    
    @contextmanager
    def validation_mode(self):
        """Context manager for validation mode"""
        self.encoder.eval()
        self.decoder.eval()
        try:
            yield
        finally:
            pass
    
    def train(self, num_epochs: int = 10, save_every: int = 1, 
              validate_every: int = 1, log_every: int = 100, shutdown_handler=None, temperature: float = 1.0):
        """Complete modern training loop"""
        
        # --- MODIFIED DataLoader creation ---
        current_batch_size = self.batch_sizer.current_batch_size
        
        # Create our custom temperature sampler
        train_sampler = TemperatureSampler(
            self.train_dataset, 
            batch_size=current_batch_size,
            temperature=temperature
        )
        
        # The DataLoader now uses our custom sampler instead of its own shuffle
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=current_batch_size,
            sampler=train_sampler, # Use the custom sampler
            num_workers=8,
            pin_memory=True,
            # Note: shuffle must be False when using a custom sampler
        )
        
        # Modern schedulers with warm restarts
        num_training_steps = len(train_loader) * num_epochs
        
        # Use modern PyTorch schedulers instead of transformers
        encoder_scheduler = CosineAnnealingWarmRestarts(
            self.encoder_optimizer,
            T_0=num_training_steps // 4,  # Restart every quarter of training
            T_mult=2,  # Double the restart period each time
            eta_min=1e-7
        )
        
        decoder_scheduler = CosineAnnealingWarmRestarts(
            self.decoder_optimizer,
            T_0=num_training_steps // 4,
            T_mult=2,
            eta_min=1e-7
        )
        
        logger.info(f"🚀 Starting training for {num_epochs} epochs")
        logger.info(f"📊 Training steps: {num_training_steps}, Batch size: {current_batch_size}")
        
        for epoch in range(num_epochs):
            # INITIALIZATION:
            epoch_loss = 0.0
            val_loss = float('inf')

            if shutdown_handler and shutdown_handler.should_stop():
                logger.info("Graceful shutdown requested")
                self._save_checkpoint(epoch, epoch_loss, val_loss)
                break

            with resource_monitor.monitor(f"epoch_{epoch}"): # monitoring resource usage
                epoch_start_time = time.time()
            
                with self.training_mode():
                    epoch_loss = self._train_epoch(
                        train_loader, 
                        encoder_scheduler, 
                        decoder_scheduler,
                        epoch,
                        log_every
                    )
            
                # Validation
                if epoch % validate_every == 0:
                    val_loss = self._validate_epoch()
                
                    # Log epoch results
                    wandb.log({
                        'epoch/train_loss': epoch_loss,
                        'epoch/val_loss': val_loss,
                        'epoch/learning_rate': encoder_scheduler.get_last_lr()[0],
                        'epoch/time': time.time() - epoch_start_time,
                        'epoch/current_batch_size': current_batch_size
                    })
                
                    logger.info(f"Epoch {epoch+1}/{num_epochs} - Train: {epoch_loss:.4f}, Val: {val_loss:.4f}")
                
                    # Adaptive batch sizing based on validation loss
                    if hasattr(self, 'prev_val_loss'):
                        if val_loss > self.prev_val_loss * 1.05:  # Loss increased by 5%
                            current_batch_size = self.batch_sizer.decrease_batch_size()
                            logger.info(f"📉 Reduced batch size to {current_batch_size}")
                        elif val_loss < self.prev_val_loss * 0.95:  # Loss decreased by 5%
                            current_batch_size = self.batch_sizer.increase_batch_size()
                            logger.info(f"📈 Increased batch size to {current_batch_size}")
                
                    self.prev_val_loss = val_loss
                else:
                    val_loss = float('inf')
            
                # Save checkpoint
                if epoch % save_every == 0:
                    self._save_checkpoint(epoch, epoch_loss, val_loss)
            
                # Clean up GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            logger.info("🎉 Training completed!")
            wandb.finish()      
        
        summary = resource_monitor.get_summary()
        logger.info(f"Training resource summary: {summary}")
        
    
    def _train_epoch(self, train_loader, encoder_scheduler, decoder_scheduler, epoch, log_every):
        """Train one epoch with modern features"""
        
        epoch_loss = 0
        step_count = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            step_start_time = time.time()
            
            # Move batch to device with non-blocking transfer
            source_ids = batch['source_ids'].to(self.device, non_blocking=True)
            target_ids = batch['target_ids'].to(self.device, non_blocking=True)
            source_mask = batch['source_mask'].to(self.device, non_blocking=True)
            target_mask = batch['target_mask'].to(self.device, non_blocking=True)
            
            # Get vocabulary info
            vocab_pack_name = batch.get('vocab_pack_name', 'default')
            pad_token_id = batch.get('pad_token_id', 0)
            
            # Forward pass with mixed precision
            if self.memory_config.mixed_precision:
                with torch.cuda.amp.autocast():
                    loss = self._forward_pass(source_ids, target_ids, source_mask, pad_token_id)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping before unscaling
                self.scaler.unscale_(self.encoder_optimizer)
                self.scaler.unscale_(self.decoder_optimizer)
                
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.0)
                
                # Optimizer steps with scaling
                self.scaler.step(self.encoder_optimizer)
                self.scaler.step(self.decoder_optimizer)
                self.scaler.update()
            else:
                # Standard precision training
                loss = self._forward_pass(source_ids, target_ids, source_mask, vocab_pack)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.0)
                
                # Optimizer steps
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()
            
            # Scheduler steps
            encoder_scheduler.step()
            decoder_scheduler.step()
            
            # Zero gradients
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            
            # Update metrics
            epoch_loss += loss.item()
            step_count += 1
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{encoder_scheduler.get_last_lr()[0]:.2e}",
                'step_time': f"{time.time() - step_start_time:.2f}s"
            })
            
            # Logging
            if self.global_step % log_every == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/encoder_lr': encoder_scheduler.get_last_lr()[0],
                    'train/decoder_lr': decoder_scheduler.get_last_lr()[0],
                    'train/step_time': time.time() - step_start_time,
                    'train/global_step': self.global_step,
                    'system/gpu_memory': torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                })
        
        return epoch_loss / step_count
    
    def _forward_pass(self, source_ids, target_ids, source_mask, pad_token_id):
        """Forward pass with modern optimizations"""
        
        # Encoder forward pass
        encoder_output = self.encoder(source_ids, source_mask)
        
        # Decoder forward pass (teacher forcing)
        decoder_output = self.decoder(
            target_ids[:, :-1],
            encoder_output,
            encoder_attention_mask=source_mask
        )
        
        # Calculate loss
        loss = nn.functional.cross_entropy(
            decoder_output.reshape(-1, decoder_output.size(-1)),
            target_ids[:, 1:].reshape(-1),
            ignore_index=pad_token_id,
            label_smoothing=0.1  # Modern label smoothing
        )
        
        return loss
    
    def _validate_epoch(self):
        """Validation with modern inference mode"""
        
        val_loader = self.trainer.create_optimized_dataloader(
            self.val_dataset,
            batch_size=self.batch_sizer.current_batch_size * 2,  # Larger batch for validation
            shuffle=False,
            num_workers=4
        )
        
        total_loss = 0
        step_count = 0
        
        with self.validation_mode():
            with torch.inference_mode():  # Modern replacement for torch.no_grad()
                for batch in tqdm(val_loader, desc="Validating"):
                    # Move batch to device
                    source_ids = batch['source_ids'].to(self.device, non_blocking=True)
                    target_ids = batch['target_ids'].to(self.device, non_blocking=True)
                    source_mask = batch['source_mask'].to(self.device, non_blocking=True)
                    
                    # Dynamic vocabulary loading
                    vocab_pack = batch['vocab_pack']
                    self.encoder.load_vocabulary_pack(vocab_pack)
                    self.decoder.load_vocabulary_pack(vocab_pack)
                    
                    # Forward pass
                    if self.memory_config.mixed_precision:
                        with torch.cuda.amp.autocast():
                            loss = self._forward_pass(source_ids, target_ids, source_mask, vocab_pack)
                    else:
                        loss = self._forward_pass(source_ids, target_ids, source_mask, vocab_pack)
                    
                    total_loss += loss.item()
                    step_count += 1
        
        return total_loss / step_count if step_count > 0 else float('inf')

    def _save_checkpoint(self, epoch, train_loss, val_loss):
        """Save checkpoint with modern format , safetensors support and versioning"""
        
        checkpoint_data = {
            'version': CHECKPOINT_VERSION,  # Add version
            'epoch': epoch,
            'global_step': self.global_step,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'encoder_optimizer': self.encoder_optimizer.state_dict(),
            'decoder_optimizer': self.decoder_optimizer.state_dict(),
            'memory_config': self.memory_config.__dict__,
            'batch_sizer_state': self.batch_sizer.get_state(),
            'pytorch_version': torch.__version__,
            'timestamp': time.time()
        }
        
        # Save model weights separately using safetensors
        encoder_path = self.checkpoint_dir / f"encoder_epoch_{epoch}.safetensors"
        decoder_path = self.checkpoint_dir / f"decoder_epoch_{epoch}.safetensors"
        
        try:
            # Save with safetensors format
            safetensors.torch.save_file(
                self.encoder.state_dict(),
                encoder_path
            )
            safetensors.torch.save_file(
                self.decoder.state_dict(),
                decoder_path
            )
            
            # Save training state as regular pickle
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint_data, checkpoint_path)
            
            logger.info(f"💾 Checkpoint saved to {checkpoint_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ Safetensors save failed: {e}, falling back to torch.save")
            
            # Fallback to regular torch.save
            checkpoint_data.update({
                'encoder_state_dict': self.encoder.state_dict(),
                'decoder_state_dict': self.decoder.state_dict(),
            })
            
            fallback_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}_fallback.pt"
            torch.save(checkpoint_data, fallback_path)
        
        # Save best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            
            try:
                safetensors.torch.save_file(
                    self.encoder.state_dict(),
                    self.checkpoint_dir / "best_encoder.safetensors"
                )
                safetensors.torch.save_file(
                    self.decoder.state_dict(),
                    self.checkpoint_dir / "best_decoder.safetensors"
                )
                
                # Save best checkpoint metadata
                best_checkpoint = checkpoint_data.copy()
                best_checkpoint['is_best'] = True
                torch.save(best_checkpoint, self.checkpoint_dir / "best_checkpoint.pt")
                
                logger.info(f"🏆 New best model saved! Val loss: {val_loss:.4f}")
                
            except Exception as e:
                logger.warning(f"⚠️ Best model save failed: {e}")

    def load_checkpoint(self, checkpoint_path: str, load_optimizer_state: bool = True):
        """Load checkpoint with modern format support and version checking"""
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Check version compatibility
        checkpoint_version = checkpoint.get('version', '1.0')
        if checkpoint_version != CHECKPOINT_VERSION:
            logger.warning(f"Checkpoint version mismatch: {checkpoint_version} vs {CHECKPOINT_VERSION}")
            # Handle migration if needed
        
        # Load model states
        if 'encoder_state_dict' in checkpoint:
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        else:
            # Try to load from separate safetensors files
            encoder_path = checkpoint_path.parent / "encoder.safetensors"
            if encoder_path.exists():
                encoder_state = safetensors.torch.load_file(encoder_path)
                self.encoder.load_state_dict(encoder_state)
        
        if 'decoder_state_dict' in checkpoint:
            self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        else:
            decoder_path = checkpoint_path.parent / "decoder.safetensors"
            if decoder_path.exists():
                decoder_state = safetensors.torch.load_file(decoder_path)
                self.decoder.load_state_dict(decoder_state)
        
        # Load training state
        if load_optimizer_state:
            if 'encoder_optimizer' in checkpoint:
                self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
            if 'decoder_optimizer' in checkpoint:
                self.decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
        
        # Restore training state
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('val_loss', float('inf'))
        
        if 'batch_sizer_state' in checkpoint:
            self.batch_sizer.load_state(checkpoint['batch_sizer_state'])
   
        logger.info(f"✅ Checkpoint loaded from {checkpoint_path}")
        logger.info(f"📊 Resumed at global step {self.global_step}")

 # Modern dataset class with improved features
class ModernParallelDataset(torch.utils.data.Dataset):
    """Modern parallel dataset with caching and preprocessing"""
    
    def __init__(self, data_path: str, cache_dir: Optional[str] =  None, vocab_dir: str = 'vocabs'):
        self.data_path = Path(data_path)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_path.parent / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load or create cached data
        self.data = self._load_or_create_cache()

        # Initialize VocabularyManager with vocab directory
        self.vocab_manager = VocabularyManager(vocab_dir=vocab_dir)
        
        logger.info(f"📚 Dataset loaded: {len(self.data)} samples")
    
    def _load_or_create_cache(self):
        """Load cached data or create cache from raw data"""
        
        cache_file = self.cache_dir / f"{self.data_path.stem}_cache.json"
        
        if cache_file.exists():
            logger.info(f"📦 Loading cached data from {cache_file}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        logger.info(f"🔄 Creating cache from {self.data_path}")
        data = self._load_raw_data()
        
        # Save cache
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return data
    
    def _load_raw_data(self):
        """Load and preprocess raw parallel data"""
        
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                try:
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        data.append({
                            'source': parts[0].strip(),
                            'target': parts[1].strip(),
                            'source_lang': parts[2].strip(),
                            'target_lang': parts[3].strip(),
                            'line_no': line_no
                        })
                except Exception as e:
                    logger.warning(f"⚠️ Error processing line {line_no}: {e}")
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get vocabulary pack for this language pair
        vocab_pack = self.vocab_manager.get_vocab_for_pair(
            item['source_lang'], 
            item['target_lang']
        )
        
        # Tokenize with modern preprocessing
        source_tokens = self._tokenize_with_special_tokens(
            item['source'], 
            item['source_lang'], 
            vocab_pack
        )
        target_tokens = self._tokenize_with_special_tokens(
            item['target'], 
            item['target_lang'], 
            vocab_pack
        )

        # Pad or truncate to max length (e.g., 512)
        max_length = 512
        source_tokens = self._pad_or_truncate(source_tokens, max_length)
        target_tokens = self._pad_or_truncate(target_tokens, max_length)
        
        # Create attention masks
        source_mask = [1 if tok != 0 else 0 for tok in source_tokens]
        target_mask = [1 if tok != 0 else 0 for tok in target_tokens]
        
        return {
            'source_ids': torch.tensor(source_tokens, dtype=torch.long),
            'target_ids': torch.tensor(target_tokens, dtype=torch.long),
            'source_mask': torch.tensor(source_mask, dtype=torch.long),
            'target_mask': torch.tensor(target_mask, dtype=torch.long),
            # Add vocab_pack info
            'vocab_pack_name': vocab_pack.name if hasattr(vocab_pack, 'name') else 'default',
            'vocab_size': vocab_pack.size if hasattr(vocab_pack, 'size') else len(vocab_pack.tokens),
            'pad_token_id': vocab_pack.special_tokens.get('<pad>', 0), 
            'unk_token_id': vocab_pack.special_tokens.get('<unk>', 1),
            'metadata': {
                'source_lang': item['source_lang'],
                'target_lang': item['target_lang'],
                'line_no': item.get('line_no', idx)
            }
        }

    def _pad_or_truncate(self, tokens: List[int], max_length: int) -> List[int]:
        """Pad or truncate tokens to max_length"""
        if len(tokens) > max_length:
            return tokens[:max_length]
        else:
            return tokens + [0] * (max_length - len(tokens))    
    
    def _tokenize_with_special_tokens(self, text: str, lang: str, vocab_pack):
        """Tokenize text with proper special token handling"""
        
        # Add language-specific special tokens
        tokens = [vocab_pack.tokens.get('<bos>', 0)]
        
        # Simple tokenization (replace with proper tokenizer)
        word_tokens = text.lower().split()
        for word in word_tokens:
            token_id = vocab_pack.tokens.get(word, vocab_pack.tokens.get('<unk>', 1))
            tokens.append(token_id)
        
        tokens.append(vocab_pack.tokens.get('<eos>', 2))
        
        return tokens

    # Add to ModernUniversalSystemTrainer
    def enable_quantization_aware_training(self, num_bits: int = 8):
        """Enable QAT for better quantization results"""
        self.qat_trainer = QuantizationAwareTrainer(num_bits)
    
        # Wrap forward pass
        original_forward = self._forward_pass
    
        def qat_forward_pass(*args, **kwargs):
            output = original_forward(*args, **kwargs)
            if self.training:
                output = self.qat_trainer.fake_quantize(output)
            return output
    
        self._forward_pass = qat_forward_pass
        logger.info(f"✅ QAT enabled with {num_bits} bits")

class EnhancedDynamicBatchSizer(BaseDynamicBatchSizer):
    """Enhanced dynamic batch sizing with memory awareness"""
    
    def __init__(self, initial_batch_size: int = 32, 
                 min_batch_size: int = 4,
                 max_batch_size: int = 128):
        super().__init__(initial_batch_size, max_batch_size)
        self.min_batch_size = min_batch_size
        self.memory_threshold = 0.9
        self.history = []
        self.adjustment_factor = 1.2
        
    def increase_batch_size(self) -> int:
        """Increase batch size within limits"""
        new_size = min(self.max_batch_size, int(self.current_batch_size * self.adjustment_factor))
        if new_size != self.current_batch_size:
            logger.info(f"🔼 Increasing batch size: {self.current_batch_size} -> {new_size}")
            self.current_batch_size = new_size
        return self.current_batch_size
    
    def decrease_batch_size(self) -> int:
        """Decrease batch size within limits"""
        new_size = max(self.min_batch_size, int(self.current_batch_size / self.adjustment_factor))
        if new_size != self.current_batch_size:
            logger.info(f"🔽 Decreasing batch size: {self.current_batch_size} -> {new_size}")
            self.current_batch_size = new_size
        return self.current_batch_size
    
    def adjust_batch_size(self, loss: float = None, 
                         memory_used: float = None) -> int:
        """Adjust based on both loss and memory"""
        
        if memory_used and memory_used > self.memory_threshold:
            # Emergency reduction
            self.current_batch_size = max(
                self.min_batch_size,
                self.current_batch_size // 2
            )
            logger.warning(f"⚠️ Memory critical! Batch size -> {self.current_batch_size}")
            return self.current_batch_size
        
        # Track loss history
        if loss is not None:
            self.history.append(loss)
            
            # Adjust based on loss trend
            if len(self.history) >= 3:
                recent_trend = self.history[-1] - self.history[-3]
                
                if recent_trend > 0:  # Loss increasing
                    self.decrease_batch_size()
                elif recent_trend < -0.01:  # Loss decreasing well
                    self.increase_batch_size()
        
        return self.current_batch_size    

class ProfileGuidedTrainer:
    """Training with profiling for optimization insights"""
    
    def __init__(self, base_trainer: ModernUniversalSystemTrainer):
        self.base_trainer = base_trainer
        self.profiling_results = {}
        self.optimization_suggestions = []
        
    def profile_training_step(self, num_steps: int = 10, 
                            trace_path: str = "./profiler_traces") -> Dict[str, Any]:
        """Profile training to find bottlenecks"""
        
        train_loader = self.base_trainer.trainer.create_optimized_dataloader(
            self.base_trainer.train_dataset, 
            batch_size=self.base_trainer.batch_sizer.current_batch_size
        )
        
        # Setup profiler with comprehensive options
        profiler_kwargs = {
            'activities': [ProfilerActivity.CPU, ProfilerActivity.CUDA],
            'record_shapes': True,
            'profile_memory': True,
            'with_stack': True,
            'on_trace_ready': tensorboard_trace_handler(trace_path),
            'schedule': torch.profiler.schedule(
                wait=1,      # Skip first step
                warmup=1,    # Warmup for 1 step
                active=num_steps - 2,  # Profile active steps
                repeat=1
            )
        }
        
        results = {
            'step_times': [],
            'memory_usage': [],
            'top_operations': [],
            'bottlenecks': []
        }
        
        logger.info(f"🔄 Profiling {num_steps} training steps...")
        
        with profile(**profiler_kwargs) as prof:
            for i, batch in enumerate(train_loader):
                if i >= num_steps:
                    break
                
                start_time = time.time()
                
                # Run training step
                loss = self.base_trainer._forward_pass(
                    batch['source_ids'].to(self.base_trainer.device),
                    batch['target_ids'].to(self.base_trainer.device),
                    batch['source_mask'].to(self.base_trainer.device),
                    batch['vocab_pack']
                )
                
                loss.backward()
                self.base_trainer.encoder_optimizer.step()
                self.base_trainer.decoder_optimizer.step()
                self.base_trainer.encoder_optimizer.zero_grad()
                self.base_trainer.decoder_optimizer.zero_grad()
                
                # Record metrics
                step_time = time.time() - start_time
                results['step_times'].append(step_time)
                
                if torch.cuda.is_available():
                    results['memory_usage'].append({
                        'allocated': torch.cuda.memory_allocated() / 1024**3,
                        'reserved': torch.cuda.memory_reserved() / 1024**3
                    })
                
                prof.step()
        
        # Analyze results
        logger.info("📊 Analyzing profiling results...")
        
        # Get key metrics
        key_averages = prof.key_averages()
        
        # Find top time-consuming operations
        top_ops = key_averages.table(sort_by="cuda_time_total", row_limit=10)
        results['top_operations'] = top_ops
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(key_averages)
        results['bottlenecks'] = bottlenecks
        
        # Generate optimization suggestions
        self.optimization_suggestions = self._generate_optimization_suggestions(results)
        
        # Save detailed report
        self._save_profiling_report(results, trace_path)
        
        # Print summary
        print("\n" + "="*60)
        print("PROFILING SUMMARY")
        print("="*60)
        print(f"Average step time: {np.mean(results['step_times']):.3f}s")
        if results['memory_usage']:
            avg_memory = np.mean([m['allocated'] for m in results['memory_usage']])
            print(f"Average GPU memory: {avg_memory:.2f}GB")
        print(f"\nTop time-consuming operations:")
        print(top_ops)
        print("\nOptimization suggestions:")
        for suggestion in self.optimization_suggestions:
            print(f"  - {suggestion}")
        print("="*60)
        
        return results
    
    def _identify_bottlenecks(self, key_averages) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks from profiling data"""
        bottlenecks = []
        
        # Analyze for common bottlenecks
        for avg in key_averages:
            # Check for excessive memory allocation
            if avg.cpu_memory_usage > 1024**3:  # > 1GB
                bottlenecks.append({
                    'type': 'memory',
                    'operation': avg.key,
                    'memory_gb': avg.cpu_memory_usage / 1024**3,
                    'suggestion': 'Consider gradient checkpointing or smaller batch size'
                })
            
            # Check for slow operations
            if avg.cuda_time_total > 1000000:  # > 1 second total
                bottlenecks.append({
                    'type': 'compute',
                    'operation': avg.key,
                    'time_ms': avg.cuda_time_total / 1000,
                    'count': avg.count,
                    'suggestion': 'Consider optimizing or replacing this operation'
                })
            
            # Check for CPU-GPU sync issues
            if 'copy' in avg.key.lower() and avg.count > 100:
                bottlenecks.append({
                    'type': 'data_transfer',
                    'operation': avg.key,
                    'count': avg.count,
                    'suggestion': 'Reduce CPU-GPU data transfers'
                })
        
        return bottlenecks
    
    def _generate_optimization_suggestions(self, results: Dict) -> List[str]:
        """Generate specific optimization suggestions based on profiling"""
        suggestions = []
        
        # Analyze step time variance
        step_times = results['step_times']
        if len(step_times) > 2:
            variance = np.std(step_times) / np.mean(step_times)
            if variance > 0.2:  # >20% variance
                suggestions.append("High step time variance detected - consider enabling CUDA graphs")
        
        # Memory usage suggestions
        if results['memory_usage']:
            max_memory = max(m['allocated'] for m in results['memory_usage'])
            if max_memory > 0.8 * torch.cuda.get_device_properties(0).total_memory / 1024**3:
                suggestions.append("High memory usage - enable gradient checkpointing")
        
        # Bottleneck-based suggestions
        for bottleneck in results['bottlenecks']:
            if bottleneck['type'] == 'data_transfer':
                suggestions.append("Enable pin_memory and non_blocking transfers")
            elif bottleneck['type'] == 'compute' and 'attention' in bottleneck['operation']:
                suggestions.append("Enable Flash Attention or use attention optimization")
        
        # Check for compilation opportunities
        if not self.base_trainer.memory_config.compile_model:
            suggestions.append("Enable torch.compile for potential speedup")
        
        return list(set(suggestions))  # Remove duplicates
    
    def _save_profiling_report(self, results: Dict, trace_path: str):
        """Save detailed profiling report"""
        report_path = Path(trace_path) / "profiling_report.json"
        report_path.parent.mkdir(exist_ok=True)
        
        # Prepare serializable report
        report = {
            'timestamp': time.time(),
            'average_step_time': float(np.mean(results['step_times'])),
            'step_times': results['step_times'],
            'memory_usage': results['memory_usage'],
            'bottlenecks': results['bottlenecks'],
            'suggestions': self.optimization_suggestions
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Saved profiling report to {report_path}")
    
    def benchmark_configurations(self, configs: List[Dict[str, Any]], 
                                num_steps: int = 20) -> pd.DataFrame:
        """Benchmark different training configurations"""
        
        results = []
        
        for config in configs:
            logger.info(f"🔄 Benchmarking config: {config.get('name', 'unnamed')}")
            
            # Apply configuration
            original_config = self._save_current_config()
            self._apply_config(config)
            
            # Run benchmark
            step_times = []
            memory_usage = []
            
            train_loader = self.base_trainer.trainer.create_optimized_dataloader(
                self.base_trainer.train_dataset,
                batch_size=config.get('batch_size', 32)
            )
            
            for i, batch in enumerate(train_loader):
                if i >= num_steps:
                    break
                
                start_time = time.time()
                start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
                # Training step
                loss = self.base_trainer._forward_pass(
                    batch['source_ids'].to(self.base_trainer.device),
                    batch['target_ids'].to(self.base_trainer.device),
                    batch['source_mask'].to(self.base_trainer.device),
                    batch['vocab_pack']
                )
                
                loss.backward()
                
                # Record metrics
                step_times.append(time.time() - start_time)
                if torch.cuda.is_available():
                    memory_usage.append((torch.cuda.memory_allocated() - start_memory) / 1024**3)
                
                # Clear gradients
                self.base_trainer.encoder_optimizer.zero_grad()
                self.base_trainer.decoder_optimizer.zero_grad()
            
            # Record results
            results.append({
                'config_name': config.get('name', 'unnamed'),
                'batch_size': config.get('batch_size', 32),
                'avg_step_time': np.mean(step_times),
                'std_step_time': np.std(step_times),
                'avg_memory_gb': np.mean(memory_usage) if memory_usage else 0,
                'samples_per_second': config.get('batch_size', 32) / np.mean(step_times),
                **{k: v for k, v in config.items() if k not in ['name', 'batch_size']}
            })
            
            # Restore original config
            self._restore_config(original_config)
        
        # Create comparison dataframe
        df = pd.DataFrame(results)
        df = df.sort_values('samples_per_second', ascending=False)
        
        # Print comparison
        print("\n" + "="*80)
        print("CONFIGURATION BENCHMARK RESULTS")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        
        # Save to file
        df.to_csv("benchmark_results.csv", index=False)
        logger.info("📄 Saved benchmark results to benchmark_results.csv")
        
        return df
    
    def _save_current_config(self) -> Dict:
        """Save current configuration"""
        return {
            'batch_size': self.base_trainer.batch_sizer.current_batch_size,
            'compile_model': self.base_trainer.memory_config.compile_model,
            'mixed_precision': self.base_trainer.memory_config.mixed_precision,
            'gradient_checkpointing': self.base_trainer.memory_config.gradient_checkpointing
        }
    
    def _apply_config(self, config: Dict):
        """Apply configuration changes"""
        if 'batch_size' in config:
            self.base_trainer.batch_sizer.current_batch_size = config['batch_size']
        if 'mixed_precision' in config:
            self.base_trainer.memory_config.mixed_precision = config['mixed_precision']
        # Add more config applications as needed
    
    def _restore_config(self, config: Dict):
        """Restore saved configuration"""
        self._apply_config(config)

# Real-time training dashboard
class TrainingDashboard:
    """Real-time training metrics visualization"""
    
    def __init__(self, trainer: ModernUniversalSystemTrainer):
        self.trainer = trainer
        self.metrics_history = defaultdict(list)
        
    def update(self, metrics: Dict[str, float]):
        """Update dashboard with new metrics"""
        for key, value in metrics.items():
            self.metrics_history[key].append(value)
        
        # Update live plot if available
        if self._has_matplotlib():
            self._update_plots()
    
    def _update_plots(self):
        """Update matplotlib plots"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss plot
        axes[0, 0].plot(self.metrics_history['train_loss'], label='Train')
        axes[0, 0].plot(self.metrics_history['val_loss'], label='Val')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        
        # Learning rate
        axes[0, 1].plot(self.metrics_history['lr'])
        axes[0, 1].set_title('Learning Rate')
        
        # GPU memory
        axes[1, 0].plot(self.metrics_history['gpu_memory'])
        axes[1, 0].set_title('GPU Memory (GB)')
        
        # Batch size
        axes[1, 1].plot(self.metrics_history['batch_size'])
        axes[1, 1].set_title('Dynamic Batch Size')
        
        plt.tight_layout()
        plt.savefig('training_dashboard.png')
        plt.close()  


class ExperimentComparator:
    """Compare multiple training runs"""
    
    def __init__(self, experiment_dirs: List[str]):
        self.experiments = self._load_experiments(experiment_dirs)
    
    def generate_comparison_report(self) -> pd.DataFrame:
        """Generate comparison of all experiments"""
        data = []
        
        for exp_name, exp_data in self.experiments.items():
            data.append({
                'experiment': exp_name,
                'best_val_loss': exp_data['best_val_loss'],
                'final_train_loss': exp_data['final_train_loss'],
                'total_time_hours': exp_data['total_time'] / 3600,
                'peak_gpu_memory': exp_data['peak_memory'],
                'config': exp_data['config']
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('best_val_loss')
        
        # Save report
        df.to_csv('experiment_comparison.csv', index=False)
        df.to_html('experiment_comparison.html', index=False)
        
        return df      

# Usage example
def main():
    """Main training function with all proper imports"""
    from utils.logging_config import setup_logging
    setup_logging(log_dir="logs/training", log_level="INFO")
    import torch
    from pathlib import Path
    from data.data_utils import ConfigManager

    data_config = ConfigManager.load_config()
    languages = data_config.get('languages')
    training_distribution = data_config.get('training_distribution')

    # Pass the context to the auto-selector
    config_path = auto_select_config(
        languages=languages,
        data_info=training_distribution
    )

    # Check if models exist, if not create them
    try:
        from encoder.universal_encoder import UniversalEncoder
        from cloud_decoder.optimized_decoder import OptimizedUniversalDecoder
    except ImportError:
        logger.error("Cannot import encoder/decoder modules. Please ensure they are in the Python path.")
        return

    # Auto-detect GPU and load config
    config_path = auto_select_config() # This will automatically find the best config file for your hardware
    config_dict = load_config(config_path) # Load the full, merged configuration

    logger.info(f"Loaded config from: {config_path}")
    logger.info(yaml.dump(config_dict, default_flow_style=False))
    logger.info(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Initialize models
    encoder = UniversalEncoder(
        max_vocab_size=50000,
        hidden_dim=config_dict.get('model', {}).get('hidden_dim', 1024),
        num_layers=config_dict.get('model', {}).get('num_layers', 6),
        num_heads=config_dict.get('model', {}).get('num_heads', 16),
        dropout=config_dict.get('model', {}).get('dropout', 0.1)
    )
    
    decoder = OptimizedUniversalDecoder(
        encoder_dim=config_dict.get('model', {}).get('hidden_dim', 1024),
        decoder_dim=config_dict.get('model', {}).get('decoder_dim', 512),
        vocab_size=50000,
        num_layers=config_dict.get('model', {}).get('decoder_layers', 6),
        num_heads=config_dict.get('model', {}).get('decoder_heads', 8),
        dropout=config_dict.get('model', {}).get('dropout', 0.1)
    )
    
    # Check if data exists
    train_data_path = Path(config_dict['data']['processed_dir']) / 'train_final.txt'
    val_data_path = Path(config_dict['data']['processed_dir']) / 'val_final.txt'
    
    if not train_data_path.exists():
        logger.error(f"Training data not found at {train_data_path}")
        logger.info("Please run the data pipeline first: python -m data.practical_data_pipeline")
        return

    # Configure memory optimizations from config
    memory_config = MemoryConfig(
        mixed_precision=config_dict["memory"].get("mixed_precision", True),
        gradient_checkpointing=config_dict["memory"].get("gradient_checkpointing", True),
        compile_model=config_dict["memory"].get("compile_model", True),
        compile_mode=config_dict["memory"].get("compile_mode", "reduce-overhead"),
        use_flash_attention=config_dict["memory"].get("use_flash_attention", True),
        dtype=getattr(torch, config_dict["memory"].get("dtype", "bfloat16")),
        use_safetensors=config_dict["memory"].get("use_safetensors", True),
        cpu_offload=config_dict["memory"].get("cpu_offload", False),
        activation_offload=config_dict["memory"].get("activation_offload", False),
    )

    # Initialize trainer
    trainer = ModernUniversalSystemTrainer(
        encoder=encoder,
        decoder=decoder,
        train_data_path=str(train_data_path),
        val_data_path=str(val_data_path),
        config=memory_config,
        experiment_name=f"universal-translation-{Path(config_path).stem}"
    )

    # Configure training parameters from config
    training_config = config_dict.get('training', {})
    
    def cleanup():
        """Emergency cleanup function"""
        if 'trainer' in locals() and hasattr(trainer, 'save_checkpoint'):
            logger.info("Saving emergency checkpoint...")
            trainer.save_checkpoint(
                epoch=trainer.current_epoch if hasattr(trainer, 'current_epoch') else 0,
                epoch_loss=0,
                val_loss=0,
                filepath="checkpoints/emergency_checkpoint.pt"
            )
    
    shutdown_handler = GracefulShutdown(cleanup_func=cleanup)

    # Start training
    trainer.train(
        num_epochs=training_config.get('num_epochs', 20),
        save_every=training_config.get('save_every', 2),
        validate_every=training_config.get('validate_every', 1),
        log_every=training_config.get('log_every', 50),
        shutdown_handler=shutdown_handler
    )
    logger.info("✅ Training completed successfully!")

    # Versioning:
    versioning = ModelVersion()
    
    # Register encoder
    encoder_version = versioning.register_model(
        model_path="models/universal_encoder.pt",
        model_type="encoder",
        metrics={
            'final_loss': trainer.best_val_loss,
            'training_time_hours': total_time / 3600
        },
        metadata={
            'config': config_dict,
            'dataset': 'universal_v1',
            'pytorch_version': torch.__version__
        }
    )
    logger.info(f"Encoder registered as version: {encoder_version}")

# QuantizationAwareTrainer fake_quantize method:
class QuantizationAwareTrainer:
    """This helps model learn to be robust to quantization!"""
    def __init__(self, quantization_bits: int = 8):
        self.quantization_aware = True
        self.quantization_bits = quantization_bits
        self.training = True
        
    def train_step(self, model, input_data):
        # Normal forward pass
        output = model(input_data)
        
        # Simulate quantization effects during training
        if self.quantization_aware:
            output = self.fake_quantize(output)
        
        return output
    
    def fake_quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Implement fake quantization for QAT"""
        if not self.quantization_aware or not self.training:
            return tensor
        
        # Get quantization parameters
        num_bits = self.quantization_bits
        qmin = -(2 ** (num_bits - 1))
        qmax = (2 ** (num_bits - 1)) - 1
        
        # Calculate scale and zero point per channel for better accuracy
        if tensor.dim() > 1:
            # Per-channel quantization
            min_vals = tensor.min(dim=0, keepdim=True)[0]
            max_vals = tensor.max(dim=0, keepdim=True)[0]
        else:
            # Per-tensor quantization
            min_vals = tensor.min()
            max_vals = tensor.max()
        
        # Calculate scale
        scale = (max_vals - min_vals) / (qmax - qmin)
        scale = torch.clamp(scale, min=1e-8)  # Prevent zero scale
        
        # Calculate zero point
        zero_point = qmin - torch.round(min_vals / scale)
        zero_point = torch.clamp(zero_point, qmin, qmax)
        
        # Quantize and dequantize
        tensor_q = torch.round(tensor / scale + zero_point)
        tensor_q = torch.clamp(tensor_q, qmin, qmax)
        tensor_dq = (tensor_q - zero_point) * scale
        
        # Straight-through estimator: gradients pass through unchanged
        return tensor + (tensor_dq - tensor).detach()    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()