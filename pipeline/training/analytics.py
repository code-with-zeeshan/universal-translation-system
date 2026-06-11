# training/training_analytics.py
"""
Training analytics for comprehensive training metrics collection.
"""

import time
import json
import numpy as np
from collections import defaultdict
from typing import Dict, Any


class TrainingAnalytics:
    """Comprehensive training analytics integrated into IntelligentTrainer"""
    
    def __init__(self, training_history: Dict):
        self.metrics = defaultdict(list)
        self.training_history = training_history  # Reference to trainer's history
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
        
        # Also update main training history
        self.training_history['gradient_norms'].append(grad_norm)
        self.training_history['memory_snapshots'].append(memory_gb)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate training analytics report"""
        return {
            'duration_hours': (time.time() - self.start_time) / 3600,
            'avg_tokens_per_sec': np.mean(self.metrics['tokens_per_sec']) if self.metrics['tokens_per_sec'] else 0,
            'peak_memory_gb': max(self.metrics['memory_gb']) if self.metrics['memory_gb'] else 0,
            'final_loss': self.metrics['loss'][-1] if self.metrics['loss'] else None,
            'loss_reduction': (
                (self.metrics['loss'][0] - self.metrics['loss'][-1]) / self.metrics['loss'][0] 
                if len(self.metrics['loss']) > 1 else 0
            ),
            'total_steps': len(self.metrics['loss']),
            'gradient_norm_stats': {
                'mean': np.mean(self.metrics['grad_norm']) if self.metrics['grad_norm'] else 0,
                'std': np.std(self.metrics['grad_norm']) if self.metrics['grad_norm'] else 0,
                'max': max(self.metrics['grad_norm']) if self.metrics['grad_norm'] else 0,
            }
        }
