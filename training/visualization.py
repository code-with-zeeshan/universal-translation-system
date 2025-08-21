# training/visualization.py
"""
Training visualization and dashboard utilities
Extracted from train_universal_system.py
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import json
import time
import seaborn as sns
sns.set_style('whitegrid')

logger = logging.getLogger(__name__)


class TrainingDashboard:
    """Real-time training metrics visualization"""
    
    def __init__(self, trainer: Any, output_dir: str = 'training_visualizations'):
        """
        Initialize dashboard with trainer instance
        
        Args:
            trainer: Any trainer instance (IntelligentTrainer, etc.)
            output_dir: Directory to save visualization outputs
        """
        self.trainer = trainer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_history = defaultdict(list)
        self.figure_cache = {}
        self.update_frequency = 10  # Update plots every N steps
        self.last_update = 0
        
        # Setup figure layout
        self._setup_figure()
        
    def _setup_figure(self):
        """Setup the main figure with subplots"""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Universal Translation System - Training Dashboard', fontsize=16)
        
        # Create subplot grid
        self.axes = {
            'loss': plt.subplot(2, 3, 1),
            'lr': plt.subplot(2, 3, 2),
            'gpu_memory': plt.subplot(2, 3, 3),
            'batch_size': plt.subplot(2, 3, 4),
            'gradient_norm': plt.subplot(2, 3, 5),
            'throughput': plt.subplot(2, 3, 6)
        }
        
        # Set titles
        self.axes['loss'].set_title('Training & Validation Loss')
        self.axes['lr'].set_title('Learning Rate Schedule')
        self.axes['gpu_memory'].set_title('GPU Memory Usage (GB)')
        self.axes['batch_size'].set_title('Dynamic Batch Size')
        self.axes['gradient_norm'].set_title('Gradient Norm')
        self.axes['throughput'].set_title('Training Throughput')
        
        plt.tight_layout()
    
    def update(self, metrics: Dict[str, float]):
        """
        Update dashboard with new metrics
        
        Args:
            metrics: Dictionary of metric names and values
        """
        # Store metrics
        for key, value in metrics.items():
            self.metrics_history[key].append(value)
        
        # Update plots if needed
        self.last_update += 1
        if self.last_update >= self.update_frequency:
            self._update_plots()
            self.last_update = 0
    
    def _update_plots(self):
        """Update all plots with latest data"""
        # Clear all axes
        for ax in self.axes.values():
            ax.clear()
        
        # Plot training and validation loss
        if 'train_loss' in self.metrics_history:
            steps = range(len(self.metrics_history['train_loss']))
            self.axes['loss'].plot(steps, self.metrics_history['train_loss'], 
                                  label='Train Loss', linewidth=2, alpha=0.8)
            
            if 'val_loss' in self.metrics_history:
                val_steps = range(0, len(self.metrics_history['val_loss']))
                self.axes['loss'].plot(val_steps, self.metrics_history['val_loss'], 
                                      label='Val Loss', linewidth=2, alpha=0.8, linestyle='--')
            
            self.axes['loss'].set_xlabel('Steps')
            self.axes['loss'].set_ylabel('Loss')
            self.axes['loss'].legend()
            self.axes['loss'].grid(True, alpha=0.3)
            
            # Add smoothed trend line
            if len(self.metrics_history['train_loss']) > 20:
                smoothed = self._smooth_curve(self.metrics_history['train_loss'])
                self.axes['loss'].plot(steps, smoothed, 'r-', alpha=0.3, linewidth=3)
        
        # Plot learning rate
        if 'lr' in self.metrics_history:
            steps = range(len(self.metrics_history['lr']))
            self.axes['lr'].plot(steps, self.metrics_history['lr'], 
                                color='orange', linewidth=2)
            self.axes['lr'].set_xlabel('Steps')
            self.axes['lr'].set_ylabel('Learning Rate')
            self.axes['lr'].set_yscale('log')
            self.axes['lr'].grid(True, alpha=0.3)
        
        # Plot GPU memory
        if 'gpu_memory' in self.metrics_history:
            steps = range(len(self.metrics_history['gpu_memory']))
            memory_values = self.metrics_history['gpu_memory']
            self.axes['gpu_memory'].plot(steps, memory_values, 
                                        color='green', linewidth=2)
            self.axes['gpu_memory'].fill_between(steps, 0, memory_values, 
                                                alpha=0.3, color='green')
            self.axes['gpu_memory'].set_xlabel('Steps')
            self.axes['gpu_memory'].set_ylabel('Memory (GB)')
            self.axes['gpu_memory'].grid(True, alpha=0.3)
            
            # Add memory limit line if available
            if hasattr(self.trainer, 'device') and self.trainer.device.type == 'cuda':
                import torch
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.axes['gpu_memory'].axhline(y=total_memory, color='red', 
                                               linestyle='--', alpha=0.5, 
                                               label=f'Total: {total_memory:.1f}GB')
                self.axes['gpu_memory'].legend()
        
        # Plot batch size
        if 'batch_size' in self.metrics_history:
            steps = range(len(self.metrics_history['batch_size']))
            self.axes['batch_size'].step(steps, self.metrics_history['batch_size'], 
                                        where='mid', color='purple', linewidth=2)
            self.axes['batch_size'].set_xlabel('Steps')
            self.axes['batch_size'].set_ylabel('Batch Size')
            self.axes['batch_size'].grid(True, alpha=0.3)
        
        # Plot gradient norm
        if 'gradient_norm' in self.metrics_history:
            steps = range(len(self.metrics_history['gradient_norm']))
            grad_norms = self.metrics_history['gradient_norm']
            self.axes['gradient_norm'].plot(steps, grad_norms, 
                                          color='red', linewidth=1, alpha=0.7)
            
            # Add rolling average
            if len(grad_norms) > 10:
                rolling_avg = self._rolling_average(grad_norms, window=10)
                self.axes['gradient_norm'].plot(steps[:len(rolling_avg)], rolling_avg, 
                                              color='darkred', linewidth=2, 
                                              label='Rolling Avg')
            
            self.axes['gradient_norm'].set_xlabel('Steps')
            self.axes['gradient_norm'].set_ylabel('Gradient Norm')
            self.axes['gradient_norm'].set_yscale('log')
            self.axes['gradient_norm'].grid(True, alpha=0.3)
            self.axes['gradient_norm'].legend()
        
        # Plot throughput
        if 'tokens_per_second' in self.metrics_history:
            steps = range(len(self.metrics_history['tokens_per_second']))
            throughput = self.metrics_history['tokens_per_second']
            self.axes['throughput'].plot(steps, throughput, 
                                        color='blue', linewidth=2, alpha=0.7)
            self.axes['throughput'].set_xlabel('Steps')
            self.axes['throughput'].set_ylabel('Tokens/Second')
            self.axes['throughput'].grid(True, alpha=0.3)
            
            # Add average line
            if throughput:
                avg_throughput = np.mean(throughput)
                self.axes['throughput'].axhline(y=avg_throughput, color='blue', 
                                               linestyle='--', alpha=0.5, 
                                               label=f'Avg: {avg_throughput:.0f}')
                self.axes['throughput'].legend()
        
        # Add titles with current values
        for key, ax in self.axes.items():
            if key in self.metrics_history and self.metrics_history[key]:
                current_value = self.metrics_history[key][-1]
                if key == 'loss':
                    ax.set_title(f'Loss (Current: {current_value:.4f})')
                elif key == 'lr':
                    ax.set_title(f'Learning Rate (Current: {current_value:.2e})')
                elif key == 'gpu_memory':
                    ax.set_title(f'GPU Memory (Current: {current_value:.2f}GB)')
                elif key == 'batch_size':
                    ax.set_title(f'Batch Size (Current: {int(current_value)})')
                elif key == 'gradient_norm':
                    ax.set_title(f'Gradient Norm (Current: {current_value:.2e})')
                elif key == 'throughput':
                    ax.set_title(f'Throughput (Current: {current_value:.0f} tok/s)')
        
        # Save figure
        plt.tight_layout()
        self.save_dashboard()
    
    def _smooth_curve(self, values: List[float], weight: float = 0.9) -> List[float]:
        """Apply exponential moving average smoothing"""
        smoothed = []
        last = values[0]
        for point in values:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed
    
    def _rolling_average(self, values: List[float], window: int = 10) -> List[float]:
        """Calculate rolling average"""
        ret = []
        for i in range(len(values) - window + 1):
            ret.append(np.mean(values[i:i+window]))
        return ret
    
    def save_dashboard(self, filename: Optional[str] = None):
        """Save dashboard to file"""
        if filename is None:
            filename = f'training_dashboard_{time.strftime("%Y%m%d_%H%M%S")}.png'
        
        filepath = self.output_dir / filename
        self.fig.savefig(filepath, dpi=100, bbox_inches='tight')
        
        # Also save as latest
        latest_path = self.output_dir / 'dashboard_latest.png'
        self.fig.savefig(latest_path, dpi=100, bbox_inches='tight')
    
    def save_metrics(self, filename: Optional[str] = None):
        """Save metrics history to JSON"""
        if filename is None:
            filename = f'metrics_{time.strftime("%Y%m%d_%H%M%S")}.json'
        
        filepath = self.output_dir / filename
        
        # Convert defaultdict to regular dict for JSON serialization
        metrics_dict = dict(self.metrics_history)
        
        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        logger.info(f"📊 Saved metrics to {filepath}")
    
    def create_summary_plots(self):
        """Create summary plots at the end of training"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Training Summary', fontsize=16)
        
        # Loss curve with phases
        if 'train_loss' in self.metrics_history:
            ax = axes[0, 0]
            train_loss = self.metrics_history['train_loss']
            steps = range(len(train_loss))
            
            ax.plot(steps, train_loss, label='Train Loss', alpha=0.7)
            
            # Mark different training phases
            if len(train_loss) > 100:
                # Warmup phase
                warmup_end = int(len(train_loss) * 0.1)
                ax.axvspan(0, warmup_end, alpha=0.2, color='yellow', label='Warmup')
                
                # Main training
                main_end = int(len(train_loss) * 0.8)
                ax.axvspan(warmup_end, main_end, alpha=0.2, color='green', label='Main')
                
                # Fine-tuning
                ax.axvspan(main_end, len(train_loss), alpha=0.2, color='blue', label='Fine-tune')
            
            ax.set_xlabel('Steps')
            ax.set_ylabel('Loss')
            ax.set_title('Training Phases')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Learning rate schedule
        if 'lr' in self.metrics_history:
            ax = axes[0, 1]
            lr = self.metrics_history['lr']
            steps = range(len(lr))
            
            ax.plot(steps, lr, color='orange', linewidth=2)
            ax.set_xlabel('Steps')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        # Memory usage histogram
        if 'gpu_memory' in self.metrics_history:
            ax = axes[1, 0]
            memory = self.metrics_history['gpu_memory']
            
            ax.hist(memory, bins=50, color='green', alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(memory), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(memory):.2f}GB')
            ax.set_xlabel('Memory (GB)')
            ax.set_ylabel('Frequency')
            ax.set_title('GPU Memory Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Gradient norm distribution
        if 'gradient_norm' in self.metrics_history:
            ax = axes[1, 1]
            grad_norms = self.metrics_history['gradient_norm']
            
            # Log scale for better visualization
            log_norms = np.log10(np.array(grad_norms) + 1e-8)
            
            ax.hist(log_norms, bins=50, color='red', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Log10(Gradient Norm)')
            ax.set_ylabel('Frequency')
            ax.set_title('Gradient Norm Distribution')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        summary_path = self.output_dir / 'training_summary.png'
        fig.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"📊 Saved training summary to {summary_path}")
    
    def _has_matplotlib(self) -> bool:
        """Check if matplotlib is available"""
        try:
            import matplotlib.pyplot as plt
            return True
        except ImportError:
            return False