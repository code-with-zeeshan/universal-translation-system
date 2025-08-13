# training/memory_efficient_training.py (memory-optimized)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import gc
import psutil
import warnings
import os
from typing import Optional, Dict, Any, Tuple, List, Callable
from dataclasses import dataclass
import logging
from contextlib import contextmanager
import time

logger = logging.getLogger(__name__)

@dataclass
class MemoryConfig:
    """Configuration for memory optimization"""
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    cpu_offload: bool = False
    activation_offload: bool = False
    use_flash_attention: bool = True
    dynamic_batch_size: bool = True
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    compile_model: bool = True
    empty_cache_freq: int = 100
    max_split_size: int = 512  # MB
    use_fused_optimizer: bool = True
    enable_nested_tensor: bool = True
    use_channels_last: bool = True
    compile_mode: str = "reduce-overhead"  # "default", "reduce-overhead", "max-autotune"
    use_inductor: bool = True
    dtype: torch.dtype = torch.bfloat16
    profile_memory: bool = False
    use_safetensors: bool = True

class MemoryOptimizedTrainer:
    """Modern memory-efficient training with latest optimizations"""
    
    def __init__(self, model: nn.Module, config: MemoryConfig = None):
        self.model = model
        self.config = config or MemoryConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Apply GPU optimizations
        from utils.gpu_utils import optimize_gpu_memory
        optimize_gpu_memory()

        self.step = 0
        self.memory_tracker = MemoryTracker() if self.config.profile_memory else None
        
        # Initialize memory optimizations
        self._setup_memory_optimizations()
        
        # Setup mixed precision with modern torch.amp
        if self.config.mixed_precision:
            self.scaler = torch.amp.GradScaler(device='cuda')
            self.autocast_ctx = torch.amp.autocast(device_type='cuda', dtype=self.config.dtype)
        else:
            self.scaler = None
            self.autocast_ctx = None
    
    def _setup_memory_optimizations(self):
        """Setup comprehensive memory optimizations"""
        
        # 1. CUDA memory management
        if torch.cuda.is_available():
            # Enable expandable segments (PyTorch 2.0+)
            torch.cuda.empty_cache() # Clear cache after GPU optimization
            
            # Set memory allocator backend if available
            if hasattr(torch.cuda, 'CUDAPluggableAllocator'):
                torch.cuda.set_allocator_settings('expandable_segments:True')
        
        # 2. Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        # 3. Setup Flash Attention
        if self.config.use_flash_attention:
            self._setup_flash_attention()
        
        # 4. Enable channels last memory format
        if self.config.use_channels_last:
            self._enable_channels_last()
        
        # 6. Enable nested tensor support
        if self.config.enable_nested_tensor:
            self._enable_nested_tensors()
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing with modern API"""
        def checkpoint_wrapper(module):
            def forward(*args, **kwargs):
                return checkpoint(module._old_forward, *args, **kwargs, use_reentrant=False)
            
            if hasattr(module, '_old_forward'):
                return  # Already wrapped
            
            module._old_forward = module.forward
            module.forward = forward
        
        # Apply to transformer layers
        for module in self.model.modules():
            if any(name in str(type(module)) for name in ['TransformerBlock', 'Block', 'Layer']):
                checkpoint_wrapper(module)
    
    def _setup_flash_attention(self):
        """Setup Flash Attention with modern PyTorch SDPA"""
        try:
            # Check if flash attention is available
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                # Enable flash attention backend
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_math_sdp(False)  # Disable slower math backend
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                
                logger.info("✅ Flash Attention enabled via PyTorch SDPA")
            else:
                logger.warning("⚠️ Flash Attention not available in this PyTorch version")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not enable Flash Attention: {e}")
    
    def _enable_channels_last(self): # This method should be called before moving model to device
        """Enable channels last memory format for better performance"""
        try:
            # Convert model to channels last
            self.model = self.model.to(memory_format=torch.channels_last)
            logger.info("✅ Channels last memory format enabled")
        except Exception as e:
            logger.warning(f"⚠️ Could not enable channels last: {e}")

    def _compile_model(self):
        """Compile model with modern torch.compile"""
        try:
            compile_kwargs = {
                'mode': self.config.compile_mode,
                'dynamic': True,
                'fullgraph': False,
                'backend': 'inductor' if self.config.use_inductor else 'aot_eager'
            }
            
            # Modern compile options
            if hasattr(torch, '_dynamo'):
                torch._dynamo.config.suppress_errors = True
                torch._dynamo.config.cache_size_limit = 512
                
            self.model = torch.compile(self.model, **compile_kwargs)
            logger.info(f"✅ Model compiled with mode: {self.config.compile_mode}")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not compile model: {e}")
    
    def _enable_nested_tensors(self):
        """Enable nested tensor support for variable length sequences with full implementation"""
        try:
            # Check PyTorch version
            pytorch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        
            if pytorch_version >= (2, 0):
                # Modern nested tensor support
                if hasattr(torch, '_nested_tensor_from_mask'):
                   # Enable nested tensors globally
                   torch._C._set_nested_tensor_enabled(True)
                
                   # Patch model forward to use nested tensors
                   self._patch_model_for_nested_tensors()
                
                   logger.info("✅ Nested tensor support enabled (PyTorch 2.0+)")
                else:
                    logger.warning("⚠️ Nested tensor API not found in this PyTorch version")
        
            elif pytorch_version >= (1, 13):
                # Legacy nested tensor support
                try:
                    import torch.nested
                    torch.nested.nested_tensor_from_mask = True
                    logger.info("✅ Legacy nested tensor support enabled (PyTorch 1.13+)")
                except ImportError:
                    logger.warning("⚠️ Nested tensor module not available")
            else:
                logger.warning(f"⚠️ PyTorch {torch.__version__} doesn't support nested tensors")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not enable nested tensors: {e}")

    def _patch_model_for_nested_tensors(self):
        """Patch model to use nested tensors for variable-length sequences"""
    
        original_forward = self.model.forward
    
        def nested_forward(input_ids, attention_mask=None, **kwargs):
            """Forward pass with nested tensor conversion"""
        
            # Convert to nested tensor if we have variable lengths
            if attention_mask is not None and self.config.enable_nested_tensor:
                try:
                    # Get actual sequence lengths
                    seq_lengths = attention_mask.sum(dim=1)
                
                    # Check if sequences have variable lengths
                    if seq_lengths.min() != seq_lengths.max():
                        # Convert to nested tensor
                        nested_input = self._create_nested_tensor(input_ids, seq_lengths)
                    
                        # Call original forward with nested tensor
                        # Note: Model needs to support nested tensors
                        return original_forward(nested_input, attention_mask=None, **kwargs)
                except Exception as e:
                    logger.debug(f"Nested tensor conversion failed: {e}, using regular tensors")
        
            # Fallback to regular forward
            return original_forward(input_ids, attention_mask=attention_mask, **kwargs)
    
        self.model.forward = nested_forward
        logger.info("✅ Model patched for nested tensor support")

    def _create_nested_tensor(self, padded_tensor: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Create nested tensor from padded tensor and lengths"""
    
        batch_size = padded_tensor.size(0)
        device = padded_tensor.device
    
        # Create list of unpadded tensors
        tensor_list = []
        for i in range(batch_size):
            length = lengths[i].item()
            unpadded = padded_tensor[i, :length]
            tensor_list.append(unpadded)
    
        # Create nested tensor
        if hasattr(torch, 'nested_tensor'):
            # Modern API
            nested = torch.nested_tensor(tensor_list, device=device)
        else:
            # Legacy API
            nested = torch._nested_tensor_from_tensor_list(
                tensor_list,
                dtype=padded_tensor.dtype,
                device=device
            )
    
        return nested

    # Add this utility method for converting back:
    def _nested_to_padded(self, nested_tensor: torch.Tensor, 
                     padding_value: int = 0,
                     max_length: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert nested tensor back to padded tensor with mask"""
    
        if not hasattr(nested_tensor, 'to_padded_tensor'):
            # Manual conversion for older versions
            tensors = nested_tensor.unbind()
        
            # Find max length
            if max_length is None:
                max_length = max(t.size(0) for t in tensors)
        
            # Create padded tensor
            batch_size = len(tensors)
            padded_shape = (batch_size, max_length) + tensors[0].shape[1:]
            padded = torch.full(padded_shape, padding_value, 
                               dtype=tensors[0].dtype, device=tensors[0].device)
        
            # Create mask
            mask = torch.zeros(batch_size, max_length, dtype=torch.bool, device=tensors[0].device)
        
            # Fill in values
            for i, t in enumerate(tensors):
                length = t.size(0)
                padded[i, :length] = t
                mask[i, :length] = True
        
            return padded, mask
        else:
            # Modern API
            padded = nested_tensor.to_padded_tensor(padding_value, max_length)
        
            # Create mask from nested tensor lengths
            lengths = nested_tensor.tensor_lengths()
            batch_size = padded.size(0)
            max_len = padded.size(1)
        
            mask = torch.arange(max_len, device=padded.device).expand(batch_size, max_len) < lengths.unsqueeze(1)
        
            return padded, mask
    
    def create_optimized_dataloader(self, dataset, batch_size: int, shuffle: bool = True,
                                  num_workers: int = None) -> DataLoader:
        """Create optimized DataLoader with modern features"""
        
        if num_workers is None:
            num_workers = min(8, os.cpu_count() or 1)
        
        dataloader_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'pin_memory': self.config.pin_memory,
            'persistent_workers': self.config.persistent_workers and num_workers > 0,
            'prefetch_factor': self.config.prefetch_factor if num_workers > 0 else None,
            'drop_last': True,  # For stable batch sizes
        }
        
        # Remove None values
        dataloader_kwargs = {k: v for k, v in dataloader_kwargs.items() if v is not None}
        
        return DataLoader(dataset, **dataloader_kwargs)
    
    def create_optimized_optimizer(self, parameters, lr: float = 1e-4, 
                                 weight_decay: float = 0.01) -> torch.optim.Optimizer:
        """Create optimized optimizer with modern features"""
        
        optimizer_kwargs = {
            'lr': lr,
            'weight_decay': weight_decay,
            'eps': 1e-8,
            'betas': (0.9, 0.95),  # Modern beta values
        }
        
        # Add fused optimizer if available
        if self.config.use_fused_optimizer and torch.cuda.is_available():
            optimizer_kwargs['fused'] = True
            
        return torch.optim.AdamW(parameters, **optimizer_kwargs)
    
    def train_step(self, batch: Dict[str, torch.Tensor], 
                   model: nn.Module, optimizer: torch.optim.Optimizer,
                   loss_fn: Callable) -> float:
        """Optimized training step with nested tensor support"""
        
        # Memory profiling
        if self.memory_tracker:
            self.memory_tracker.start_step()

        # Convert batch to nested tensors if enabled
        if self.config.enable_nested_tensor and 'attention_mask' in batch:
            batch = self._convert_batch_to_nested(batch)
        
        # Move batch to device with non_blocking
        batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Convert to channels last if enabled
        if self.config.use_channels_last:
            batch = {k: v.to(memory_format=torch.channels_last) if isinstance(v, torch.Tensor) and v.dim() == 4 else v 
                    for k, v in batch.items()}
        
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        # Forward pass with autocast
        if self.config.mixed_precision:
            with self.autocast_ctx:
                loss = loss_fn(model, batch)
        else:
            loss = loss_fn(model, batch)
        
        # Backward pass
        if self.config.mixed_precision:
            self.scaler.scale(loss).backward()
            
            # Gradient clipping with scaler
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step with scaler
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Memory management
        self.step += 1
        if self.step % self.config.empty_cache_freq == 0:
            self._cleanup_memory()
        
        if self.memory_tracker:
            self.memory_tracker.end_step()
        
        return loss.item()

    def _convert_batch_to_nested(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert batch to use nested tensors where applicable"""
    
        converted_batch = batch.copy()
    
        # Check if we have variable length sequences
        if 'attention_mask' in batch:
            seq_lengths = batch['attention_mask'].sum(dim=1)
        
            if seq_lengths.min() != seq_lengths.max():
                # Convert input_ids to nested tensor
                for key in ['input_ids', 'labels', 'decoder_input_ids']:
                    if key in batch:
                        converted_batch[key] = self._create_nested_tensor(batch[key], seq_lengths)

                # Remove attention mask as it's encoded in nested tensor
                if 'attention_mask' in converted_batch:
                    del converted_batch['attention_mask']
            
                logger.debug("Converted batch to nested tensors")
    
        return converted_batch       
    
    @torch.inference_mode()  # Modern replacement for torch.no_grad()
    def validation_step(self, batch: Dict[str, torch.Tensor],
                       model: nn.Module, loss_fn: Callable) -> float:
        """Optimized validation step"""
        
        # Move batch to device
        batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Convert to channels last if enabled
        if self.config.use_channels_last:
            batch = {k: v.to(memory_format=torch.channels_last) if isinstance(v, torch.Tensor) and v.dim() == 4 else v 
                    for k, v in batch.items()}
        
        # Forward pass with autocast
        if self.config.mixed_precision:
            with self.autocast_ctx:
                loss = loss_fn(model, batch)
        else:
            loss = loss_fn(model, batch)
        
        return loss.item()
    
    def _cleanup_memory(self):
        """Advanced memory cleanup"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Force garbage collection
            gc.collect()
            
            # Log memory usage
            if self.memory_tracker:
                self.memory_tracker.log_memory_usage()
    
    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                       scheduler, epoch: int, loss: float, filepath: str):
        """Save checkpoint with modern safetensors support"""
        
        # Get model state dict (handle compiled models)
        if hasattr(model, '_orig_mod'):
            model_state = model._orig_mod.state_dict()
        else:
            model_state = model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': loss,
            'config': self.config,
            'pytorch_version': torch.__version__,
        }
        
        # Save with safetensors if available
        if self.config.use_safetensors:
            try:
                from safetensors.torch import save_file
                # Save model separately with safetensors
                save_file(model_state, filepath.replace('.pt', '_model.safetensors'))
                # Save other data with torch
                other_data = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
                torch.save(other_data, filepath.replace('.pt', '_metadata.pt'))
                logger.info(f"✅ Saved checkpoint with safetensors: {filepath}")
            except ImportError:
                torch.save(checkpoint, filepath)
                logger.info(f"✅ Saved checkpoint: {filepath}")
        else:
            torch.save(checkpoint, filepath)
            logger.info(f"✅ Saved checkpoint: {filepath}")
    
    def load_checkpoint(self, filepath: str, model: nn.Module, 
                       optimizer: torch.optim.Optimizer = None,
                       scheduler = None) -> Dict[str, Any]:
        """Load checkpoint with safetensors support"""
        
        if self.config.use_safetensors and os.path.exists(filepath.replace('.pt', '_model.safetensors')):
            try:
                from safetensors.torch import load_file
                # Load model state
                model_state = load_file(filepath.replace('.pt', '_model.safetensors'))
                # Load metadata
                metadata = torch.load(filepath.replace('.pt', '_metadata.pt'), map_location=self.device)
                
                # Combine
                checkpoint = {**metadata, 'model_state_dict': model_state}
                logger.info(f"✅ Loaded checkpoint with safetensors: {filepath}")
                
            except ImportError:
                checkpoint = torch.load(filepath, map_location=self.device)
                logger.info(f"✅ Loaded checkpoint: {filepath}")
        else:
            checkpoint = torch.load(filepath, map_location=self.device)
            logger.info(f"✅ Loaded checkpoint: {filepath}")
        
        # Load states
        if hasattr(model, '_orig_mod'):
            model._orig_mod.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint
    
    @contextmanager
    def memory_profiler(self):
        """Context manager for memory profiling"""
        if self.memory_tracker:
            self.memory_tracker.start_profiling()
            try:
                yield
            finally:
                self.memory_tracker.stop_profiling()
        else:
            yield

    # In MemoryOptimizedTrainer
    def _select_amp_dtype(self) -> torch.dtype:
        """Automatically select best AMP dtype"""
        if not torch.cuda.is_available():
            return torch.float32
        
        # Check GPU capabilities
        device_capability = torch.cuda.get_device_capability()
    
        # Use BF16 for newer GPUs (A100, H100)
        if device_capability[0] >= 8:
            if torch.cuda.is_bf16_supported():
                logger.info("Using BFloat16 for AMP (Ampere or newer GPU)")
                return torch.bfloat16
    
        # Use FP16 for older GPUs
        logger.info("Using Float16 for AMP")
        return torch.float16        


class MemoryTracker:
    """Advanced memory tracking and profiling"""
    
    def __init__(self):
        self.start_time = None
        self.step_times = []
        self.memory_usage = []
        self.peak_memory = 0
        
    def start_step(self):
        """Start tracking a training step"""
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def end_step(self):
        """End tracking a training step"""
        if self.start_time:
            step_time = time.time() - self.start_time
            self.step_times.append(step_time)
            
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
                self.memory_usage.append(current_memory)
                self.peak_memory = max(self.peak_memory, peak_memory)
    
    def log_memory_usage(self):
        """Log current memory usage"""
        if torch.cuda.is_available():
            current = torch.cuda.memory_allocated() / 1024**3
            peak = torch.cuda.max_memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            
            logger.info(f"📊 Memory - Current: {current:.2f}GB, Peak: {peak:.2f}GB, Reserved: {reserved:.2f}GB")
    
    def start_profiling(self):
        """Start comprehensive profiling"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.start_time = time.time()
    
    def stop_profiling(self):
        """Stop profiling and generate report"""
        if self.start_time:
            total_time = time.time() - self.start_time
            
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / 1024**3
                
                logger.info(f"📈 Profile Report:")
                logger.info(f"   Total time: {total_time:.2f}s")
                logger.info(f"   Peak memory: {peak_memory:.2f}GB")
                logger.info(f"   Average step time: {sum(self.step_times)/len(self.step_times):.4f}s")


class DynamicBatchSizer:
    """Dynamic batch sizing based on memory usage"""
    
    def __init__(self, initial_batch_size: int = 32, max_batch_size: int = 128):
        self.current_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = 8
        self.memory_threshold = 0.9  # 90% of GPU memory
        self.adjustment_factor = 1.2
        
    def adjust_batch_size(self) -> int:
        """Adjust batch size based on memory usage"""
        if not torch.cuda.is_available():
            return self.current_batch_size
        
        memory_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        
        if memory_usage > self.memory_threshold:
            # Reduce batch size
            new_size = max(self.min_batch_size, int(self.current_batch_size / self.adjustment_factor))
            if new_size != self.current_batch_size:
                logger.info(f"🔽 Reducing batch size: {self.current_batch_size} -> {new_size}")
                self.current_batch_size = new_size
        elif memory_usage < 0.7:
            # Increase batch size
            new_size = min(self.max_batch_size, int(self.current_batch_size * self.adjustment_factor))
            if new_size != self.current_batch_size:
                logger.info(f"🔼 Increasing batch size: {self.current_batch_size} -> {new_size}")
                self.current_batch_size = new_size
        
        return self.current_batch_size


# Example usage and utilities
def create_modern_training_setup(model: nn.Module, dataset, 
                               config: MemoryConfig = None) -> Dict[str, Any]:
    """Create a complete modern training setup"""
    
    config = config or MemoryConfig()
    trainer = MemoryOptimizedTrainer(model, config)
    
    # Create optimized components
    optimizer = trainer.create_optimized_optimizer(model.parameters())
    
    # Modern scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1000, T_mult=2, eta_min=1e-7
    )
    
    # Dynamic batch sizer
    batch_sizer = DynamicBatchSizer()
    
    return {
        'trainer': trainer,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'batch_sizer': batch_sizer,
        'config': config
    }


def benchmark_training_speed(model: nn.Module, dataset, config: MemoryConfig = None):
    """Benchmark training speed with different optimizations"""
    
    results = {}
    configs = [
        ("Baseline", MemoryConfig(gradient_checkpointing=False, mixed_precision=False, compile_model=False)),
        ("Mixed Precision", MemoryConfig(gradient_checkpointing=False, mixed_precision=True, compile_model=False)),
        ("Compiled", MemoryConfig(gradient_checkpointing=False, mixed_precision=False, compile_model=True)),
        ("Full Optimization", MemoryConfig()),
    ]
    
    for name, test_config in configs:
        logger.info(f"🔄 Benchmarking: {name}")
        
        trainer = MemoryOptimizedTrainer(model, test_config)
        optimizer = trainer.create_optimized_optimizer(model.parameters())
        
        # Simple benchmark
        start_time = time.time()
        for i, batch in enumerate(dataset):
            if i >= 10:  # Only test 10 batches
                break
            trainer.train_step(batch, model, optimizer, lambda m, b: torch.randn(1, requires_grad=True))
        
        end_time = time.time()
        results[name] = end_time - start_time
        logger.info(f"✅ {name}: {results[name]:.2f}s")
    
    return results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create a simple model for testing
    model = nn.Sequential(
        nn.Linear(1000, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    # Create modern training setup
    config = MemoryConfig(
        gradient_checkpointing=True,
        mixed_precision=True,
        compile_model=True,
        use_flash_attention=True
    )
    
    setup = create_modern_training_setup(model, None, config)
    trainer = setup['trainer']
    
    logger.info("🚀 Modern memory-efficient training setup complete!")
    logger.info(f"📊 Configuration: {config}")