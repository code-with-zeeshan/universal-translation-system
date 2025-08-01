# training/training_utils.py
"""
Common utilities for training modules
"""
import torch
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import json
from pathlib import Path
from utils.exceptions import TrainingError

logger = logging.getLogger(__name__)

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
                
        except TrainingError as e:
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