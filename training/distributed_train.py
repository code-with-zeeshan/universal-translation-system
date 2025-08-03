# training/distributed_train.py (multi-GPU)
import torch
import argparse
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from pathlib import Path
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
    LocalStateDictConfig,
    OptimStateDictConfig,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
    size_based_auto_wrap_policy,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
import torch.nn.functional as F
import os
import logging
from typing import Optional, Dict, Any, Union, List, Tuple
from contextlib import contextmanager
import functools
import time
import psutil
import gc
import socket
from utils.exceptions import TrainingError
from dataclasses import dataclass
from torch.profiler import profile, record_function, ProfilerActivity
from collections import defaultdict
from utils.shutdown_handler import GracefulShutdown
from utils.model_versioning import ModelVersion
from utils.resource_monitor import resource_monitor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration dataclass
@dataclass
class TrainingConfig:
    """Unified training configuration"""
    use_fsdp: bool = True
    mixed_precision: bool = True
    cpu_offload: bool = False
    gradient_checkpointing: bool = True
    activation_checkpointing: bool = True 
    compile_model: bool = True
    compile_mode: str = "max-autotune" 
    flash_attention: bool = True
    use_fp8: bool = False  # Future feature
    accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    save_every: int = 1000
    log_every: int = 100
    profile_training: bool = False  
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    backward_prefetch: BackwardPrefetch = BackwardPrefetch.BACKWARD_PRE

# Port finding
def find_free_port():
    """Find a free port for distributed training"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

# Environment setup
def setup_distributed_environment():
    """Setup environment variables for distributed training"""
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = str(find_free_port())
    
    # Enable optimizations
    os.environ['TORCH_COMPILE_DEBUG'] = '0'
    os.environ['TORCH_LOGS'] = '+dynamo'
    os.environ['TORCHDYNAMO_DISABLE_CACHE_LIMIT'] = '1'
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

class UnifiedDistributedTrainer:
    """Unified distributed training with best features"""
    
    def __init__(self, 
                 gpu_id: int, 
                 world_size: int,
                 config: TrainingConfig = None):
        
        self.gpu_id = gpu_id
        self.world_size = world_size
        self.config = config or TrainingConfig()
        
        # Initialize distributed training
        self._init_distributed()
        
        # Set device
        torch.cuda.set_device(gpu_id)
        self.device = torch.device(f'cuda:{gpu_id}')

        # Optimize GPU memory
        from utils.gpu_utils import optimize_gpu_memory
        optimize_gpu_memory()
        
        # Initialize modern mixed precision scaler with BFloat16 support 
        if self.config.mixed_precision:
            self.scaler = torch.amp.GradScaler(device='cuda')
            self.use_bfloat16 = torch.cuda.is_bf16_supported()
            self.mixed_precision_dtype = torch.bfloat16 if self.use_bfloat16 else torch.float16
        else:
            self.scaler = None
            self.use_bfloat16 = False
            self.mixed_precision_dtype = torch.float32
        
        # Performance monitoring 
        self.step_times = []
        self.memory_usage = []
        self.step = 0
        self.accumulated_loss = 0.0  
        
        # Setup Flash Attention with better error handling 
        if self.config.flash_attention:
            self._setup_flash_attention()
        
        logger.info(f"Initialized unified trainer on GPU {gpu_id}/{world_size}")
        logger.info(f"Using {'FSDP' if self.config.use_fsdp else 'DDP'} with compile={self.config.compile_model}")
        logger.info(f"Mixed precision: {self.config.mixed_precision} (dtype: {self.mixed_precision_dtype})")
        logger.info(f"Flash Attention: {self.config.flash_attention}")
    
    def _init_distributed(self):
        """Initialize distributed training with proper error handling and optimizations"""
        try:
            # Set environment variables for optimal performance 
            os.environ.setdefault('NCCL_TREE_THRESHOLD', '0')
            os.environ.setdefault('NCCL_IB_DISABLE', '1')
            os.environ.setdefault('NCCL_SOCKET_IFNAME', 'lo')
            
            # Initialize process group with modern timeout 
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=self.world_size,
                rank=self.gpu_id,
                timeout=torch.distributed.timedelta(seconds=3600)  # Modern API
            )
            
            # Set CUDA device
            torch.cuda.set_device(self.gpu_id)
            
            # Enable modern optimizations
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False  
            
            # Enable compile optimizations 
            if self.config.compile_model:
                torch._dynamo.config.cache_size_limit = 1024
                torch._dynamo.config.optimize_ddp = True
            
            # Clear cache at start
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Failed to initialize distributed training: {e}")
            raise
    
    def _setup_flash_attention(self):
        """Enhanced Flash Attention setup"""
        try:
            # Enable Flash Attention in PyTorch
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)  # Disable slower fallback
            
            # Try to import flash_attn for additional optimizations
            try:
                import flash_attn
                self.flash_attn_available = True
                logger.info("Flash Attention 2 available")
            except ImportError:
                self.flash_attn_available = False
                logger.info("Using PyTorch native Flash Attention")
                
        except Exception as e:
            logger.warning(f"Flash Attention setup failed: {e}")
            self.config.flash_attention = False
    
    def setup_model(self, 
                   encoder: torch.nn.Module, 
                   decoder: torch.nn.Module,
                   auto_wrap_policy: Optional[callable] = None) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """Setup models with modern distributed training and optimizations"""
        
        # Move models to GPU first
        encoder = encoder.to(self.device)
        decoder = decoder.to(self.device)
        
        # Apply gradient checkpointing (cleaner approach)
        if self.config.gradient_checkpointing:
            if hasattr(encoder, 'gradient_checkpointing_enable'):
                encoder.gradient_checkpointing_enable()
            if hasattr(decoder, 'gradient_checkpointing_enable'):
                decoder.gradient_checkpointing_enable()
        
        # Apply activation checkpointing 
        if self.config.activation_checkpointing:
            encoder = self._apply_activation_checkpointing(encoder)
            decoder = self._apply_activation_checkpointing(decoder)
        
        if self.config.use_fsdp:
            # Modern FSDP setup
            encoder = self._setup_fsdp_model(encoder, "encoder", auto_wrap_policy)
            decoder = self._setup_fsdp_model(decoder, "decoder", auto_wrap_policy)
        else:
            # Modern DDP setup with optimizations
            encoder = self._setup_ddp_model(encoder, "encoder")
            decoder = self._setup_ddp_model(decoder, "decoder")
        
        # Apply torch.compile with better mode 
        if self.config.compile_model and hasattr(torch, 'compile'):
            logger.info(f"Applying torch.compile with mode={self.config.compile_mode}")
            try:
                encoder = torch.compile(
                    encoder, 
                    mode=self.config.compile_mode,
                    fullgraph=False,
                    dynamic=True
                )
                decoder = torch.compile(
                    decoder, 
                    mode=self.config.compile_mode,
                    fullgraph=False,
                    dynamic=True
                )
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}, falling back to eager mode")
                self.config.compile_model = False
        
        return encoder, decoder
    
    def _apply_activation_checkpointing(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply activation checkpointing to transformer layers"""
        def check_fn(submodule):
            return (
                hasattr(submodule, 'self_attn') or
                hasattr(submodule, 'attention') or
                submodule.__class__.__name__ in ['TransformerBlock', 'TransformerLayer']
            )
        
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.REENTRANT
            ),
            check_fn=check_fn
        )
        
        logger.info("Applied activation checkpointing to model")
        return model
    
    def _setup_fsdp_model(self, 
                         model: torch.nn.Module, 
                         model_name: str,
                         auto_wrap_policy: Optional[callable] = None) -> FSDP:
        """Setup FSDP with unified configuration"""
        
        # Configure mixed precision with BFloat16 support 
        if self.config.mixed_precision:
            mixed_precision_policy = MixedPrecision(
                param_dtype=self.mixed_precision_dtype,
                reduce_dtype=self.mixed_precision_dtype,
                buffer_dtype=self.mixed_precision_dtype,
                cast_forward_inputs=True,
                cast_root_forward_inputs=True
            )
        else:
            mixed_precision_policy = None
        
        # Configure CPU offload
        cpu_offload_policy = CPUOffload(offload_params=True) if self.config.cpu_offload else None
        
        # Auto wrap policy
        if auto_wrap_policy is None:
            auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={
                    torch.nn.TransformerEncoderLayer,
                    torch.nn.TransformerDecoderLayer,
                }
            )
        
        # Create FSDP model with all features
        fsdp_model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision_policy,
            cpu_offload=cpu_offload_policy,
            backward_prefetch=self.config.backward_prefetch,
            sharding_strategy=self.config.sharding_strategy,
            device_id=self.gpu_id,
            sync_module_states=True,
            param_init_fn=None,
            forward_prefetch=True,
            use_orig_params=True,
            ignored_modules=None,
            process_group=None,
            limit_all_gathers=True
        )
        
        logger.info(f"Setup FSDP for {model_name} with strategy={self.config.sharding_strategy}")
        logger.info(f"Using dtype={self.mixed_precision_dtype}")
        return fsdp_model
    
    def _setup_ddp_model(self, model: torch.nn.Module, model_name: str) -> DDP:
        """Setup DDP with modern optimizations"""
        ddp_model = DDP(
            model,
            device_ids=[self.gpu_id],
            output_device=self.gpu_id,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True,
            broadcast_buffers=False,
            bucket_cap_mb=25,
        )
        
        logger.info(f"Setup DDP for {model_name} with static graph optimization")
        return ddp_model
    
    def train_step(self, 
                  batch: Dict[str, torch.Tensor],
                  encoder: torch.nn.Module,
                  decoder: torch.nn.Module,
                  optimizer: torch.optim.Optimizer,
                  scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> float:
        """Unified training step with best features"""
        
        start_time = time.time()
        
        # Enable gradient synchronization only on accumulation boundary
        should_sync = (self.step + 1) % self.config.accumulation_steps == 0
        sync_context = self._get_sync_context(encoder, decoder, should_sync)
        
        with sync_context:
            # Modern mixed precision training with BFloat16 support
            if self.config.mixed_precision:
                with torch.amp.autocast(device_type='cuda', dtype=self.mixed_precision_dtype):
                    loss = self.compute_loss(batch, encoder, decoder)
                    loss = loss / self.config.accumulation_steps
                
                # Scale loss for mixed precision
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                loss = self.compute_loss(batch, encoder, decoder)
                loss = loss / self.config.accumulation_steps
                loss.backward()
        
        # Accumulate loss 
        self.accumulated_loss += loss.item()
        
        # Update optimizer on accumulation boundary
        if should_sync:
            if self.scaler:
                # Modern mixed precision optimizer step
                self.scaler.unscale_(optimizer)
                
                # Gradient clipping after unscaling
                self._clip_gradients(encoder, decoder)
                
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # Standard optimizer step
                self._clip_gradients(encoder, decoder)
                optimizer.step()
            
            # Update scheduler
            if scheduler:
                scheduler.step()
            
            optimizer.zero_grad(set_to_none=True)
            
            # Return accumulated loss and reset
            final_loss = self.accumulated_loss
            self.accumulated_loss = 0.0
            
            # Performance monitoring
            step_time = time.time() - start_time
            self.step_times.append(step_time)
            
            # Memory monitoring
            if self.gpu_id == 0 and self.step % 100 == 0:
                self._log_memory_usage()
            
            return final_loss
        
        return 0.0  # Return 0 when not syncing
    
    def _clip_gradients(self, encoder: torch.nn.Module, decoder: torch.nn.Module):
        """Clip gradients with proper handling for FSDP"""
        if self.config.use_fsdp:
            encoder.clip_grad_norm_(self.config.max_grad_norm)
            decoder.clip_grad_norm_(self.config.max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()),
                max_norm=self.config.max_grad_norm
            )
    
    @contextmanager
    def _get_sync_context(self, encoder, decoder, should_sync):
        """Get appropriate synchronization context"""
        if self.config.use_fsdp:
            yield
        else:
            if should_sync:
                yield
            else:
                with encoder.no_sync(), decoder.no_sync():
                    yield
    
    def compute_loss(self, 
                    batch: Dict[str, torch.Tensor],
                    encoder: torch.nn.Module,
                    decoder: torch.nn.Module) -> torch.Tensor:
        """Compute loss with Flash Attention and profiling"""
        
        # Move batch to device with non-blocking transfer
        source_ids = batch['source_ids'].to(self.device, non_blocking=True)
        target_ids = batch['target_ids'].to(self.device, non_blocking=True)
        source_mask = batch['source_mask'].to(self.device, non_blocking=True)
        
        # Forward pass with Flash Attention and profiling 
        if self.config.flash_attention:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=True
            ):
                with record_function("encoder_forward"):
                    encoder_output = encoder(source_ids, source_mask)
                
                with record_function("decoder_forward"):
                    decoder_output = decoder(
                        target_ids[:, :-1],
                        encoder_output,
                        encoder_attention_mask=source_mask
                    )
        else:
            with record_function("encoder_forward"):
                encoder_output = encoder(source_ids, source_mask)
            
            with record_function("decoder_forward"):
                decoder_output = decoder(
                    target_ids[:, :-1],
                    encoder_output,
                    encoder_attention_mask=source_mask
                )
        
        # Compute loss with modern optimizations
        with record_function("loss_computation"):
            loss = F.cross_entropy(
                decoder_output.reshape(-1, decoder_output.size(-1)),
                target_ids[:, 1:].reshape(-1),
                ignore_index=0,
                label_smoothing=0.1,
                reduction='mean'
            )
        
        return loss
    
    def save_checkpoint(self, 
                       epoch: int,
                       encoder: torch.nn.Module,
                       decoder: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       save_path: str = "checkpoint.pt",
                       save_optimizer: bool = True) -> None:
        """Save checkpoint with all features"""
        
        if self.gpu_id == 0:  # Only save on main process
            checkpoint = {
                'epoch': epoch,
                'step': self.step,
                'config': self.config,  
                'torch_version': torch.__version__,
                'world_size': self.world_size,
                'model_config': {  
                    'use_fsdp': self.config.use_fsdp,
                    'mixed_precision': self.config.mixed_precision,
                    'mixed_precision_dtype': str(self.mixed_precision_dtype),
                    'sharding_strategy': self.config.sharding_strategy.name if self.config.use_fsdp else None,
                    'compile_model': self.config.compile_model,
                    'compile_mode': self.config.compile_mode
                }
            }
            
            # Save model states
            if self.config.use_fsdp:
                # FSDP state dict handling
                with FSDP.state_dict_type(
                    encoder, 
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                ):
                    checkpoint['encoder_state_dict'] = encoder.state_dict()
                
                with FSDP.state_dict_type(
                    decoder,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                ):
                    checkpoint['decoder_state_dict'] = decoder.state_dict()
                
                # Save optimizer state with FSDP
                if save_optimizer:
                    with FSDP.state_dict_type(
                        encoder,
                        StateDictType.FULL_STATE_DICT,
                        FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                    ):
                        optim_state = FSDP.optim_state_dict(encoder, optimizer)
                        checkpoint['optimizer_state_dict'] = optim_state
            else:
                # DDP state dict handling
                checkpoint['encoder_state_dict'] = encoder.module.state_dict()
                checkpoint['decoder_state_dict'] = decoder.module.state_dict()
                
                if save_optimizer:
                    checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
            # Save scheduler and scaler
            if scheduler:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            if self.scaler:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
            # Save performance metrics 
            checkpoint['performance_metrics'] = {
                'avg_step_time': sum(self.step_times) / len(self.step_times) if self.step_times else 0,
                'memory_usage': self.memory_usage[-10:] if self.memory_usage else []
            }
            
            # Use safetensors for better performance 
            try:
                import safetensors.torch as st
                st.save_file(checkpoint, save_path.replace('.pt', '.safetensors'))
                logger.info(f"Checkpoint saved to {save_path.replace('.pt', '.safetensors')} (safetensors)")
            except ImportError:
                torch.save(checkpoint, save_path)
                logger.info(f"Checkpoint saved to {save_path} (pickle)")
    
    def load_checkpoint(self, 
                       checkpoint_path: str,
                       encoder: torch.nn.Module,
                       decoder: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       load_optimizer: bool = True) -> Tuple[int, int]:
        """Load checkpoint with proper device mapping and return epoch and step"""
        
        # Load checkpoint with safetensors support
        if checkpoint_path.endswith('.safetensors'):
            try:
                import safetensors.torch as st
                checkpoint = st.load_file(checkpoint_path, device=f'cuda:{self.gpu_id}')
                logger.info("Loaded checkpoint from safetensors format")
            except ImportError:
                checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{self.gpu_id}')
        else:
            checkpoint = torch.load(
                checkpoint_path, 
                map_location=f'cuda:{self.gpu_id}',
                weights_only=False
            )
        
        # Load model states
        if self.config.use_fsdp:
            with FSDP.state_dict_type(
                encoder,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=False, rank0_only=False)
            ):
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
            
            with FSDP.state_dict_type(
                decoder,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=False, rank0_only=False)
            ):
                decoder.load_state_dict(checkpoint['decoder_state_dict'])
            
            # Load optimizer with FSDP
            if load_optimizer and 'optimizer_state_dict' in checkpoint:
                optim_state = FSDP.optim_state_dict_to_load(
                    model=encoder,
                    optim=optimizer,
                    optim_state_dict=checkpoint['optimizer_state_dict']
                )
                optimizer.load_state_dict(optim_state)
        else:
            encoder.module.load_state_dict(checkpoint['encoder_state_dict'])
            decoder.module.load_state_dict(checkpoint['decoder_state_dict'])
            
            if load_optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler and scaler
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        step = checkpoint.get('step', 0)
        self.step = step
        
        logger.info(f"Loaded checkpoint from epoch {epoch}, step {step}")
        return epoch, step
    
    def _log_memory_usage(self):
        """Log memory usage statistics"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(self.gpu_id) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(self.gpu_id) / 1024**3
            
            cpu_memory = psutil.virtual_memory().percent
            
            self.memory_usage.append({
                'gpu_allocated': memory_allocated,
                'gpu_reserved': memory_reserved,
                'cpu_memory': cpu_memory
            })
            
            logger.info(f"Memory - GPU: {memory_allocated:.2f}GB allocated, "
                       f"{memory_reserved:.2f}GB reserved, CPU: {cpu_memory:.1f}%")
    
    def cleanup(self):
        """Cleanup distributed training and clear cache"""
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        # Run garbage collection
        gc.collect()

    def validate_checkpoint(self, checkpoint_path: str) -> bool:
        """Validate checkpoint compatibility"""
        try:
            if checkpoint_path.endswith('.safetensors'):
                import safetensors.torch as st
                metadata = st.load_file(checkpoint_path, device='cpu')
            else:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            
            # Check required keys
            required_keys = ['model_state_dict', 'config', 'torch_version']
            if not all(key in checkpoint for key in required_keys):
                logger.error(f"Missing required keys in checkpoint")
                return False
            
            # Check version compatibility
            saved_version = checkpoint.get('torch_version', '0.0.0')
            current_version = torch.__version__
        
            if saved_version.split('.')[:2] != current_version.split('.')[:2]:
                logger.warning(f"Version mismatch: saved with {saved_version}, current {current_version}")
            
            return True
        
        except Exception as e:
            logger.error(f"Invalid checkpoint: {e}")
            return False    

# Add to UnifiedDistributedTrainer
class TrainingAnalytics:
    """Comprehensive training analytics"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = time.time()
    
    def log_step(self, loss: float, lr: float, grad_norm: float, 
                 memory_gb: float, tokens_per_sec: float):
        """Log training step metrics"""
        self.metrics['loss'].append(loss)
        self.metrics['lr'].append(lr)
        self.metrics['grad_norm'].append(grad_norm)
        self.metrics['memory_gb'].append(memory_gb)
        self.metrics['tokens_per_sec'].append(tokens_per_sec)
        self.metrics['timestamp'].append(time.time() - self.start_time)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate training report"""
        return {
            'duration_hours': (time.time() - self.start_time) / 3600,
            'avg_tokens_per_sec': np.mean(self.metrics['tokens_per_sec']),
            'peak_memory_gb': max(self.metrics['memory_gb']),
            'final_loss': self.metrics['loss'][-1] if self.metrics['loss'] else None,
            'loss_reduction': (self.metrics['loss'][0] - self.metrics['loss'][-1]) / self.metrics['loss'][0] if len(self.metrics['loss']) > 1 else 0
        }

# Modern training function with all optimizations
def train_with_unified_distributed(gpu_id: int, 
                                 world_size: int,
                                 encoder: torch.nn.Module,
                                 decoder: torch.nn.Module,
                                 train_loader: torch.utils.data.DataLoader,
                                 val_loader: Optional[torch.utils.data.DataLoader] = None,
                                 num_epochs: int = 10,
                                 config: TrainingConfig = None) -> None:
    """Complete unified distributed training function"""
    
    # Initialize trainer
    trainer = UnifiedDistributedTrainer(
        gpu_id=gpu_id,
        world_size=world_size,
        config=config or TrainingConfig()
    )
    
    # Setup models
    encoder, decoder = trainer.setup_model(encoder, decoder)
    
    # Modern optimizer with fused operations
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=trainer.config.lr,
        weight_decay=trainer.config.weight_decay,
        eps=1e-8,
        betas=(0.9, 0.98),
        fused=torch.cuda.is_available()
    )
    
    # Modern learning rate scheduler
    total_steps = len(train_loader) * num_epochs // trainer.config.accumulation_steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=trainer.config.lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos',
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25.0,
        final_div_factor=1e4
    )
    
    # Profiler setup
    profiler = None
    if trainer.config.profile_training and gpu_id == 0:
        profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
    
    # Training loop with modern features
    start_time = time.time()
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        
        # Set epoch for distributed sampler
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        epoch_loss = 0.0
        num_updates = 0
        
        for step, batch in enumerate(train_loader):
            trainer.step = step
            
            # Start profiling
            if profiler:
                profiler.step()
            
            # Training step
            loss = trainer.train_step(batch, encoder, decoder, optimizer, scheduler)
            
            # Only accumulate loss when we actually update
            if loss > 0:
                epoch_loss += loss
                num_updates += 1
                
                # Logging
                if num_updates % trainer.config.log_every == 0 and gpu_id == 0:
                    elapsed = time.time() - start_time
                    current_lr = scheduler.get_last_lr()[0]
                    logger.info(
                        f"Epoch {epoch}, Step {step}, Updates {num_updates}, "
                        f"Loss: {loss:.4f}, LR: {current_lr:.2e}, Time: {elapsed:.1f}s"
                    )
        
        # Validation with modern inference mode
        if val_loader and gpu_id == 0:
            val_loss = validate_model(encoder, decoder, val_loader, trainer.device, trainer.config)
            logger.info(f"Epoch {epoch} - Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trainer.save_checkpoint(
                    epoch, encoder, decoder, optimizer, scheduler,
                    "best_model.pt"
                )
        
        # Save checkpoint
        if gpu_id == 0 and (epoch + 1) % trainer.config.save_every == 0:
            trainer.save_checkpoint(
                epoch, encoder, decoder, optimizer, scheduler,
                f"checkpoint_epoch_{epoch}.pt"
            )
        
        # Synchronize all processes
        if dist.is_initialized():
            dist.barrier()
        
        avg_loss = epoch_loss / max(num_updates, 1)
        if gpu_id == 0:
            logger.info(f"Epoch {epoch} completed - Avg Loss: {avg_loss:.4f}")
    
    # Final save
    if gpu_id == 0:
        trainer.save_checkpoint(
            num_epochs - 1, encoder, decoder, optimizer, scheduler,
            "final_model.pt"
        )
    
    # Cleanup
    if profiler:
        profiler.stop()
    trainer.cleanup()

@torch.inference_mode()  # more efficient than torch.no_grad()
def validate_model(encoder: torch.nn.Module,
                  decoder: torch.nn.Module,
                  val_loader: torch.utils.data.DataLoader,
                  device: torch.device,
                  config: TrainingConfig) -> float:
    """Modern validation function with inference mode"""
    
    encoder.eval()
    decoder.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    for batch in val_loader:
        source_ids = batch['source_ids'].to(device, non_blocking=True)
        target_ids = batch['target_ids'].to(device, non_blocking=True)
        source_mask = batch['source_mask'].to(device, non_blocking=True)
        
        # Forward pass with Flash Attention if enabled
        if config.flash_attention:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=True
            ):
                encoder_output = encoder(source_ids, source_mask)
                decoder_output = decoder(
                    target_ids[:, :-1],
                    encoder_output,
                    encoder_attention_mask=source_mask
                )
        else:
            encoder_output = encoder(source_ids, source_mask)
            decoder_output = decoder(
                target_ids[:, :-1],
                encoder_output,
                encoder_attention_mask=source_mask
            )
        
        loss = F.cross_entropy(
            decoder_output.reshape(-1, decoder_output.size(-1)),
            target_ids[:, 1:].reshape(-1),
            ignore_index=0,
            reduction='mean'
        )
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


@torch.inference_mode()
def distributed_validate(encoder: torch.nn.Module,
                        decoder: torch.nn.Module,
                        val_loader: torch.utils.data.DataLoader,
                        device: torch.device,
                        config: TrainingConfig,
                        world_size: int) -> float:
    """Distributed validation with proper reduction"""
    encoder.eval()
    decoder.eval()
    
    local_loss = 0.0
    local_batches = 0
    
    for batch in val_loader:
        # ... compute loss ...
        local_loss += loss.item()
        local_batches += 1
    
    # Reduce across all processes
    if dist.is_initialized():
        loss_tensor = torch.tensor([local_loss, local_batches], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        
        total_loss = loss_tensor[0].item()
        total_batches = loss_tensor[1].item()
    else:
        total_loss = local_loss
        total_batches = local_batches
    
    return total_loss / max(total_batches, 1)    

def main():
    """
    Main entry point for orchestrating distributed training.
    This function handles setup, configuration, and spawning of training processes.
    """
    # 1. --- SETUP PHASE ---
    parser = argparse.ArgumentParser(description="Unified Distributed Training")
    parser.add_argument('--config', type=str, default=None,
                        help="Path to training config file. If not provided, auto-detects based on GPU.")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs.")
    parser.add_argument('--global-batch-size', type=int, default=128,
                        help="Total batch size across all GPUs.")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Path to a checkpoint to resume training from.")
    args = parser.parse_args()

    # Setup logging and performance optimizations first
    from utils.logging_config import setup_logging
    setup_logging(log_dir="logs/distributed_training", log_level="INFO")
    
    from utils.performance_setup import setup_performance_optimizations
    setup_performance_optimizations()

    # Setup distributed environment variables
    setup_distributed_environment()

    # Load configuration
    config_path = args.config or auto_select_config()
    try:
        config_dict = load_config(config_path)
        logger.info(f"Loaded configuration from: {config_path}")
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)

    # Check for available GPUs
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise TrainingError("No CUDA devices found. Distributed training requires GPUs.")
    logger.info(f"Found {world_size} GPUs. Starting distributed training...")

    # 2. --- RESOURCE PREPARATION PHASE ---
    try:
        # Import models and dataset (place imports here to avoid issues with multiprocessing)
        from encoder.universal_encoder import UniversalEncoder
        from cloud_decoder.optimized_decoder import OptimizedUniversalDecoder
        from training.train_universal_system import ModernParallelDataset

        # Get the model config section from the dictionary
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

        # Load datasets
        train_data_path = Path(config_dict['data']['processed_dir']) / 'train_final.txt'
        val_data_path = Path(config_dict['data']['processed_dir']) / 'val_final.txt'
        train_dataset = ModernParallelDataset(str(train_data_path))
        val_dataset = ModernParallelDataset(str(val_data_path))

    except (ImportError, FileNotFoundError) as e:
        logger.error(f"Failed to prepare resources: {e}")
        logger.error("Please ensure models are importable and data pipeline has been run.")
        sys.exit(1)

    # 3. --- EXECUTION PHASE ---
    # Setup graceful shutdown and model versioning in the main process
    def emergency_cleanup():
        logger.critical("Emergency cleanup initiated. A final checkpoint might be saved by the training process.")
        # Note: The actual checkpoint saving is handled in the child process.
        # This function is for cleaning up main process resources if any.

    shutdown_handler = GracefulShutdown(cleanup_func=emergency_cleanup)
    versioning = ModelVersion(model_dir=config_dict.get('model_dir', 'models'))

    # Prepare arguments for the spawned processes
    spawn_args = (
        world_size,
        encoder,
        decoder,
        train_dataset,
        val_dataset,
        args,
        config_dict,
        shutdown_handler
    )

    # Spawn training processes
    mp.spawn(
        train_with_unified_distributed_wrapper,
        args=spawn_args,
        nprocs=world_size,
        join=True
    )

    # 4. --- POST-TRAINING PHASE ---
    logger.info("🎉 Distributed training has completed.")
    
    # Log final resource usage summary
    summary = resource_monitor.get_summary()
    logger.info(f"Final resource usage summary: {summary}")

    # Register the final model version
    final_model_path = Path(config_dict.get('checkpoint_dir', 'checkpoints')) / "final_model.pt"
    if final_model_path.exists():
        logger.info("Registering final model version...")
        try:
            version = versioning.register_model(
                model_path=str(final_model_path),
                model_type="universal-encoder-decoder-fsdp",
                metrics={"final_loss": "N/A"}, # You can load metrics from the final checkpoint
                metadata={
                    "config_file": config_path,
                    "world_size": world_size,
                    "epochs": args.epochs
                }
            )
            logger.info(f"✅ Final model registered as version: {version}")
        except Exception as e:
            logger.error(f"Failed to register final model: {e}")


def train_with_unified_distributed_wrapper(gpu_id: int, world_size: int, encoder, decoder,
                                           train_dataset, val_dataset, args, config_dict, shutdown_handler):
    """
    Wrapper function to prepare dataloaders inside each spawned process
    before calling the main training logic.
    """
    # Calculate batch size per GPU
    batch_size_per_gpu = args.global_batch_size // world_size
    if args.global_batch_size % world_size != 0:
        logger.warning(f"Global batch size {args.global_batch_size} is not divisible by world size {world_size}. "
                       f"This may lead to uneven batches.")

    # Create dataloaders with DistributedSampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=gpu_id, shuffle=True, seed=42)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=gpu_id, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size_per_gpu, sampler=train_sampler,
        num_workers=4, pin_memory=True, drop_last=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size_per_gpu * 2, sampler=val_sampler,
        num_workers=4, pin_memory=True, drop_last=False
    )

    # Create TrainingConfig from the loaded dictionary
    training_config = TrainingConfig(**config_dict.get('training', {}))

    # Call the actual training function
    train_with_unified_distributed(
        gpu_id, world_size, encoder, decoder,
        train_loader, val_loader, args.epochs, training_config, shutdown_handler
    )


if __name__ == "__main__":
    # This is crucial for CUDA multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()