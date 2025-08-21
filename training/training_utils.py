# training/training_utils.py (ENHANCED VERSION)
"""
Common utilities for training modules - Enhanced with more shared functions
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import json
from pathlib import Path
from utils.exceptions import TrainingError
from abc import ABC, abstractmethod
from contextlib import contextmanager
import time
import wandb
from config.schemas import RootConfig
from collections import defaultdict

logger = logging.getLogger(__name__)

class BaseTrainer(ABC):
    """Base class for all trainers, containing shared logic."""

    def __init__(self, encoder, decoder, train_data_path, val_data_path, config: RootConfig, experiment_name: str):
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.experiment_name = experiment_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.checkpoint_dir = Path(self.config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        self._setup_models()
        self._setup_wandb()

    @abstractmethod
    def _setup_models(self):
        """Setup and wrap models for training (e.g., DDP, FSDP, or single GPU)."""
        pass

    def _setup_wandb(self):
        """Setup wandb with comprehensive config"""
        if self.config.monitoring.use_wandb:
            wandb.init(
                project="universal-translation", 
                name=f"{self.experiment_name}-{time.strftime('%Y%m%d-%H%M%S')}",
                config=self.config.dict(),
                tags=["modern", "refactored", "universal-system"]
            )

    @abstractmethod
    def train(self, num_epochs: int):
        """Main training loop."""
        pass

    @abstractmethod
    def _train_epoch(self, epoch: int, train_loader):
        """Logic for a single training epoch."""
        pass

    @abstractmethod
    def _validate_epoch(self, val_loader):
        """Logic for a single validation epoch."""
        pass

    def _forward_pass(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass to compute the loss."""
        source_ids = batch['source_ids'].to(self.device, non_blocking=True)
        target_ids = batch['target_ids'].to(self.device, non_blocking=True)
        source_mask = batch['source_mask'].to(self.device, non_blocking=True)
        pad_token_id = batch.get('pad_token_id', 0)

        encoder_output = self.encoder(source_ids, source_mask)
        decoder_output = self.decoder(target_ids[:, :-1], encoder_output, encoder_attention_mask=source_mask)
        
        loss = torch.nn.functional.cross_entropy(
            decoder_output.reshape(-1, decoder_output.size(-1)),
            target_ids[:, 1:].reshape(-1),
            ignore_index=pad_token_id,
            label_smoothing=0.1
        )
        return loss

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint."""
        # Implemented in subclasses to handle model-specific state dicts (e.g., FSDP)
        pass

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        # Implemented in subclasses
        pass

# ============ NEW ENHANCED UTILITIES ============

def create_optimizer_with_param_groups(
    model: Union[nn.Module, List[nn.Module]], 
    config: RootConfig,
    custom_groups: Optional[Dict[str, Dict]] = None
) -> torch.optim.Optimizer:
    """
    Create optimizer with different learning rates for different layers.
    Supports both single model and encoder-decoder pairs.
    
    Args:
        model: Single model or list of models [encoder, decoder]
        config: Training configuration
        custom_groups: Optional custom parameter groups
        
    Returns:
        Configured optimizer with parameter groups
    """
    param_groups = []
    
    # Handle single model or list of models
    models = [model] if isinstance(model, nn.Module) else model
    
    for model_idx, current_model in enumerate(models):
        model_name = ["encoder", "decoder"][model_idx] if len(models) > 1 else "model"
        
        # Default groups: embeddings, transformer layers, output layers
        embedding_params = []
        transformer_params = []
        output_params = []
        other_params = []
        
        for name, param in current_model.named_parameters():
            if not param.requires_grad:
                continue
                
            # Categorize parameters
            if 'embed' in name.lower() or 'emb' in name.lower():
                embedding_params.append(param)
            elif 'transformer' in name.lower() or 'attention' in name.lower() or 'layer' in name.lower():
                transformer_params.append(param)
            elif 'output' in name.lower() or 'lm_head' in name.lower() or 'classifier' in name.lower():
                output_params.append(param)
            else:
                other_params.append(param)
        
        # Create parameter groups with different learning rates
        base_lr = config.training.learning_rate
        
        if embedding_params:
            param_groups.append({
                'params': embedding_params,
                'lr': base_lr * 0.5,  # Lower LR for embeddings
                'name': f'{model_name}_embeddings'
            })
        
        if transformer_params:
            param_groups.append({
                'params': transformer_params,
                'lr': base_lr,
                'name': f'{model_name}_transformer'
            })
        
        if output_params:
            param_groups.append({
                'params': output_params,
                'lr': base_lr * 1.5,  # Higher LR for output layer
                'name': f'{model_name}_output'
            })
        
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': base_lr,
                'name': f'{model_name}_other'
            })
    
    # Apply custom groups if provided
    if custom_groups:
        for group_name, group_config in custom_groups.items():
            param_groups.append({
                'params': group_config['params'],
                'lr': group_config.get('lr', base_lr),
                'weight_decay': group_config.get('weight_decay', 0.01),
                'name': group_name
            })
    
    # Log parameter groups
    for group in param_groups:
        num_params = sum(p.numel() for p in group['params'])
        logger.info(f"Parameter group '{group['name']}': {num_params:,} parameters, LR={group['lr']:.2e}")
    
    # Create optimizer
    optimizer_class = getattr(torch.optim, config.training.optimizer_type, torch.optim.AdamW)
    
    optimizer = optimizer_class(
        param_groups,
        weight_decay=config.training.weight_decay,
        eps=1e-8,
        betas=(0.9, 0.95) if optimizer_class == torch.optim.AdamW else (0.9, 0.999)
    )
    
    return optimizer


def calculate_gradient_norm(
    model: Union[nn.Module, List[nn.Module]], 
    norm_type: float = 2.0,
    per_layer: bool = False
) -> Union[float, Dict[str, float]]:
    """
    Calculate total gradient norm for monitoring training stability.
    
    Args:
        model: Model or list of models
        norm_type: Type of norm (default: 2.0 for L2 norm)
        per_layer: If True, return per-layer gradient norms
        
    Returns:
        Total gradient norm or dict of per-layer norms
    """
    models = [model] if isinstance(model, nn.Module) else model
    
    if per_layer:
        gradient_norms = {}
        
        for model_idx, current_model in enumerate(models):
            model_name = ["encoder", "decoder"][model_idx] if len(models) > 1 else "model"
            
            for name, param in current_model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(norm_type).item()
                    gradient_norms[f"{model_name}.{name}"] = param_norm
        
        return gradient_norms
    else:
        # Calculate total norm
        total_norm = 0.0
        
        for current_model in models:
            parameters = [p for p in current_model.parameters() if p.grad is not None]
            if parameters:
                device = parameters[0].grad.device
                total_norm_tensor = torch.norm(
                    torch.stack([
                        torch.norm(p.grad.detach(), norm_type).to(device) 
                        for p in parameters
                    ]), 
                    norm_type
                )
                total_norm += total_norm_tensor.item()
        
        return total_norm


def get_training_diagnostics(trainer: Any) -> Dict[str, Any]:
    """
    Get detailed training diagnostics for debugging and monitoring.
    
    Args:
        trainer: Trainer instance (IntelligentTrainer or any trainer with standard attributes)
        
    Returns:
        Comprehensive diagnostics dictionary
    """
    diagnostics = {
        'timestamp': time.time(),
        'training_state': {},
        'model_state': {},
        'optimizer_state': {},
        'memory_state': {},
        'performance_metrics': {},
        'convergence_indicators': {}
    }
    
    # Training state
    diagnostics['training_state'] = {
        'current_epoch': getattr(trainer, 'current_epoch', None),
        'global_step': getattr(trainer, 'global_step', None),
        'best_val_loss': getattr(trainer, 'best_val_loss', None),
        'current_batch_size': trainer.batch_sizer.current_batch_size if hasattr(trainer, 'batch_sizer') else None,
        'accumulation_steps': trainer.strategy.accumulation_steps if hasattr(trainer, 'strategy') else None,
    }
    
    # Model state
    if hasattr(trainer, 'encoder') and hasattr(trainer, 'decoder'):
        encoder_params = sum(p.numel() for p in trainer.encoder.parameters())
        decoder_params = sum(p.numel() for p in trainer.decoder.parameters())
        
        diagnostics['model_state'] = {
            'encoder_params': encoder_params,
            'decoder_params': decoder_params,
            'total_params': encoder_params + decoder_params,
            'encoder_trainable': sum(p.numel() for p in trainer.encoder.parameters() if p.requires_grad),
            'decoder_trainable': sum(p.numel() for p in trainer.decoder.parameters() if p.requires_grad),
        }
        
        # Calculate gradient norms
        grad_norm = calculate_gradient_norm([trainer.encoder, trainer.decoder])
        diagnostics['model_state']['gradient_norm'] = grad_norm
    
    # Optimizer state
    if hasattr(trainer, 'optimizer'):
        diagnostics['optimizer_state'] = {
            'learning_rate': trainer.optimizer.param_groups[0]['lr'],
            'num_param_groups': len(trainer.optimizer.param_groups),
        }
        
        # Get per-group LRs
        for idx, group in enumerate(trainer.optimizer.param_groups):
            group_name = group.get('name', f'group_{idx}')
            diagnostics['optimizer_state'][f'lr_{group_name}'] = group['lr']
    
    # Memory state
    if torch.cuda.is_available():
        diagnostics['memory_state'] = {
            'gpu_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
            'gpu_reserved_gb': torch.cuda.memory_reserved() / 1024**3,
            'gpu_max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
            'gpu_utilization_percent': (torch.cuda.memory_allocated() / torch.cuda.max_memory_reserved()) * 100 if torch.cuda.max_memory_reserved() > 0 else 0,
        }
    
    # Performance metrics
    if hasattr(trainer, 'training_history'):
        history = trainer.training_history
        
        # Calculate recent metrics
        if 'train_loss' in history and len(history['train_loss']) > 0:
            recent_losses = history['train_loss'][-10:] if len(history['train_loss']) >= 10 else history['train_loss']
            diagnostics['performance_metrics'] = {
                'recent_train_loss_mean': np.mean(recent_losses),
                'recent_train_loss_std': np.std(recent_losses),
                'loss_trend': 'decreasing' if len(recent_losses) > 1 and recent_losses[-1] < recent_losses[0] else 'increasing',
            }
        
        # Training speed
        if 'step_times' in history and len(history['step_times']) > 0:
            recent_times = history['step_times'][-100:] if len(history['step_times']) >= 100 else history['step_times']
            diagnostics['performance_metrics'].update({
                'avg_step_time': np.mean(recent_times),
                'steps_per_second': 1.0 / np.mean(recent_times) if np.mean(recent_times) > 0 else 0,
                'estimated_time_remaining': None  # Can calculate based on remaining epochs
            })
    
    # Convergence indicators
    if hasattr(trainer, 'training_history') and 'train_loss' in trainer.training_history:
        losses = trainer.training_history['train_loss']
        
        diagnostics['convergence_indicators'] = {
            'has_converged': check_convergence(losses),
            'convergence_step': find_convergence_step(losses),
            'loss_variance': np.var(losses[-100:]) if len(losses) >= 100 else np.var(losses),
            'training_stable': np.var(losses[-100:]) < 0.01 if len(losses) >= 100 else False,
        }
    
    return diagnostics


def get_adaptive_gradient_clipping_value(
    gradient_history: List[float],
    percentile: float = 98.0,
    min_clip: float = 0.1,
    max_clip: float = 10.0
) -> float:
    """
    Calculate adaptive gradient clipping value based on gradient history.
    
    Args:
        gradient_history: List of recent gradient norms
        percentile: Percentile to use for clipping (default: 98)
        min_clip: Minimum clipping value
        max_clip: Maximum clipping value
        
    Returns:
        Adaptive clipping value
    """
    if len(gradient_history) < 10:
        return 1.0  # Default value
    
    clip_value = np.percentile(gradient_history, percentile)
    return np.clip(clip_value, min_clip, max_clip)


def analyze_loss_landscape(
    model: nn.Module,
    loss_fn: callable,
    data_batch: Dict[str, torch.Tensor],
    epsilon: float = 0.01,
    num_directions: int = 10
) -> Dict[str, float]:
    """
    Analyze loss landscape around current parameters for training stability.
    
    Args:
        model: Model to analyze
        loss_fn: Loss function
        data_batch: Single batch of data
        epsilon: Perturbation magnitude
        num_directions: Number of random directions to sample
        
    Returns:
        Loss landscape statistics
    """
    original_loss = loss_fn(model, data_batch).item()
    losses = [original_loss]
    
    # Sample random directions
    for _ in range(num_directions):
        # Save original parameters
        original_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Apply random perturbation
        for param in model.parameters():
            if param.requires_grad:
                perturbation = torch.randn_like(param) * epsilon
                param.data.add_(perturbation)
        
        # Calculate loss
        with torch.no_grad():
            perturbed_loss = loss_fn(model, data_batch).item()
            losses.append(perturbed_loss)
        
        # Restore parameters
        for name, param in model.named_parameters():
            param.data.copy_(original_params[name])
    
    return {
        'original_loss': original_loss,
        'mean_perturbed_loss': np.mean(losses[1:]),
        'std_perturbed_loss': np.std(losses[1:]),
        'max_perturbed_loss': np.max(losses[1:]),
        'min_perturbed_loss': np.min(losses[1:]),
        'landscape_smoothness': np.std(losses[1:]) / (original_loss + 1e-8),  # Lower is smoother
    }


def create_learning_rate_finder(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: callable,
    device: torch.device,
    min_lr: float = 1e-7,
    max_lr: float = 10,
    num_steps: int = 100
) -> Tuple[List[float], List[float]]:
    """
    Learning rate finder to determine optimal learning rate.
    
    Returns:
        Tuple of (learning_rates, losses)
    """
    model.train()
    learning_rates = []
    losses = []
    
    # Exponential LR schedule
    gamma = (max_lr / min_lr) ** (1 / num_steps)
    lr = min_lr
    
    data_iter = iter(train_loader)
    
    for step in range(num_steps):
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        # Set learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Forward pass
        optimizer.zero_grad()
        loss = loss_fn(model, batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Record
        learning_rates.append(lr)
        losses.append(loss.item())
        
        # Update LR
        lr *= gamma
        
        # Stop if loss explodes
        if loss.item() > losses[0] * 4:
            break
    
    return learning_rates, losses    

def get_optimal_batch_size(model: torch.nn.Module, 
                          device: torch.device,
                          test_batch_sizes: List[int] = None) -> int:
    """Automatically determine optimal batch size based on available memory"""
    if test_batch_sizes is None:
        test_batch_sizes = [8, 16, 32, 64, 128, 256]
    
    optimal_batch_size = 8  # Default fallback
    
    for batch_size in test_batch_sizes:
        try:
            # Create dummy input
            dummy_input = torch.randn(batch_size, 512).to(device)
            
            # Forward pass
            with torch.no_grad():
                _ = model(dummy_input)
            
            # If successful, try next size
            optimal_batch_size = batch_size
            
            # Clear cache
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                
        except Exception as e:
            if "out of memory" in str(e):
                logger.info(f"Batch size {batch_size} too large, using {optimal_batch_size}")
                break
            else:
                raise e
    
    return optimal_batch_size

def create_training_report(training_history: Dict[str, List[float]], 
                          output_path: str = "training_report.json"):
    """Create comprehensive training report"""
    report = {
        'final_metrics': {
            'final_loss': training_history['loss'][-1] if training_history.get('loss') else None,
            'best_loss': min(training_history['loss']) if training_history.get('loss') else None,
            'total_steps': len(training_history.get('loss', [])),
        },
        'convergence': {
            'converged': check_convergence(training_history.get('loss', [])),
            'convergence_step': find_convergence_step(training_history.get('loss', []))
        },
        'performance': {
            'avg_step_time': np.mean(training_history.get('step_times', [])),
            'total_training_time': sum(training_history.get('step_times', [])),
        },
        'full_history': training_history
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Training report saved to {output_path}")
    return report

def check_convergence(losses: List[float], window: int = 100, threshold: float = 0.001) -> bool:
    """Check if training has converged"""
    if len(losses) < window * 2:
        return False
    
    recent_mean = np.mean(losses[-window:])
    previous_mean = np.mean(losses[-2*window:-window])
    
    return abs(recent_mean - previous_mean) < threshold

def find_convergence_step(losses: List[float], window: int = 100, threshold: float = 0.001) -> Optional[int]:
    """Find the step where training converged"""
    for i in range(window * 2, len(losses)):
        if check_convergence(losses[:i], window, threshold):
            return i
    return None

def get_learning_rate_schedule(optimizer: torch.optim.Optimizer) -> List[float]:
    """Extract learning rate schedule from optimizer"""
    lrs = []
    for param_group in optimizer.param_groups:
        lrs.append(param_group['lr'])
    return lrs

def save_training_state(state: Dict[str, Any], filepath: str, use_safetensors: bool = True):
    """Save training state with multiple format support"""
    filepath = Path(filepath)
    
    if use_safetensors:
        try:
            from safetensors.torch import save_file
            # Separate model weights from other data
            model_data = {k: v for k, v in state.items() if 'state_dict' in k}
            other_data = {k: v for k, v in state.items() if 'state_dict' not in k}
            
            # Save model data with safetensors
            save_file(model_data, filepath.with_suffix('.safetensors'))
            
            # Save other data with pickle
            torch.save(other_data, filepath.with_suffix('.metadata'))
            
            logger.info(f"Saved training state with safetensors: {filepath}")
        except ImportError:
            torch.save(state, filepath)
            logger.info(f"Saved training state with pickle: {filepath}")
    else:
        torch.save(state, filepath)