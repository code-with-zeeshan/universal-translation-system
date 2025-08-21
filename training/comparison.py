# training/comparison.py
"""
Experiment comparison utilities for analyzing multiple training runs
Extracted from train_universal_system.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import torch

logger = logging.getLogger(__name__)


class ExperimentComparator:
    """Compare multiple training runs for analysis and selection"""
    
    def __init__(self, experiment_dirs: List[str], output_dir: str = 'comparisons'):
        """
        Initialize comparator with experiment directories
        
        Args:
            experiment_dirs: List of paths to experiment directories
            output_dir: Directory to save comparison outputs
        """
        self.experiment_dirs = [Path(d) for d in experiment_dirs]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiments = self._load_experiments(self.experiment_dirs)
        
        logger.info(f"📊 Loaded {len(self.experiments)} experiments for comparison")
    
    def _load_experiments(self, experiment_dirs: List[Path]) -> Dict[str, Dict]:
        """Load experiment data from directories"""
        experiments = {}
        
        for exp_dir in experiment_dirs:
            if not exp_dir.exists():
                logger.warning(f"Experiment directory not found: {exp_dir}")
                continue
            
            exp_name = exp_dir.name
            exp_data = {
                'path': exp_dir,
                'metrics': {},
                'config': {},
                'checkpoints': []
            }
            
            # Load training metrics
            metrics_file = exp_dir / 'metrics.json'
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    exp_data['metrics'] = json.load(f)
            
            # Load configuration
            config_file = exp_dir / 'config.json'
            if config_file.exists():
                with open(config_file, 'r') as f:
                    exp_data['config'] = json.load(f)
            
            # Find checkpoints
            checkpoints = list(exp_dir.glob('*.pt')) + list(exp_dir.glob('*.safetensors'))
            exp_data['checkpoints'] = sorted(checkpoints, key=lambda x: x.stat().st_mtime)
            
            # Load best metrics if available
            best_checkpoint = exp_dir / 'best_checkpoint.pt'
            if best_checkpoint.exists():
                try:
                    checkpoint = torch.load(best_checkpoint, map_location='cpu')
                    exp_data['best_val_loss'] = checkpoint.get('val_loss', float('inf'))
                    exp_data['best_epoch'] = checkpoint.get('epoch', -1)
                except Exception as e:
                    logger.warning(f"Could not load best checkpoint for {exp_name}: {e}")
            
            # Calculate additional metrics
            if 'train_loss' in exp_data['metrics']:
                train_losses = exp_data['metrics']['train_loss']
                exp_data['final_train_loss'] = train_losses[-1] if train_losses else None
                exp_data['min_train_loss'] = min(train_losses) if train_losses else None
            
            if 'val_loss' in exp_data['metrics']:
                val_losses = exp_data['metrics']['val_loss']
                exp_data['final_val_loss'] = val_losses[-1] if val_losses else None
                exp_data['min_val_loss'] = min(val_losses) if val_losses else None
            
            # Training time
            if 'step_times' in exp_data['metrics']:
                step_times = exp_data['metrics']['step_times']
                exp_data['total_time'] = sum(step_times) if step_times else 0
                exp_data['avg_step_time'] = np.mean(step_times) if step_times else 0
            
            # Memory usage
            if 'gpu_memory' in exp_data['metrics']:
                memory_usage = exp_data['metrics']['gpu_memory']
                exp_data['peak_memory'] = max(memory_usage) if memory_usage else 0
                exp_data['avg_memory'] = np.mean(memory_usage) if memory_usage else 0
            
            experiments[exp_name] = exp_data
        
        return experiments
    
    def generate_comparison_report(self) -> pd.DataFrame:
        """Generate comprehensive comparison report"""
        data = []
        
        for exp_name, exp_data in self.experiments.items():
            row = {
                'experiment': exp_name,
                'best_val_loss': exp_data.get('best_val_loss', float('inf')),
                'final_train_loss': exp_data.get('final_train_loss'),
                'min_train_loss': exp_data.get('min_train_loss'),
                'final_val_loss': exp_data.get('final_val_loss'),
                'min_val_loss': exp_data.get('min_val_loss'),
                'best_epoch': exp_data.get('best_epoch', -1),
                'total_time_hours': exp_data.get('total_time', 0) / 3600,
                'avg_step_time_s': exp_data.get('avg_step_time', 0),
                'peak_gpu_memory_gb': exp_data.get('peak_memory', 0),
                'avg_gpu_memory_gb': exp_data.get('avg_memory', 0),
                'num_checkpoints': len(exp_data['checkpoints']),
                'config_hash': hash(str(exp_data['config']))
            }
            
            # Add config details
            if exp_data['config']:
                config = exp_data['config']
                row.update({
                    'batch_size': config.get('training', {}).get('batch_size'),
                    'learning_rate': config.get('training', {}).get('learning_rate'),
                    'num_epochs': config.get('training', {}).get('num_epochs'),
                    'optimizer': config.get('training', {}).get('optimizer_type'),
                    'mixed_precision': config.get('training', {}).get('mixed_precision'),
                    'gradient_checkpointing': config.get('memory', {}).get('gradient_checkpointing'),
                })
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df = df.sort_values('best_val_loss')
        
        # Save reports
        csv_path = self.output_dir / 'experiment_comparison.csv'
        df.to_csv(csv_path, index=False)
        
        html_path = self.output_dir / 'experiment_comparison.html'
        df.to_html(html_path, index=False)
        
        logger.info(f"📊 Saved comparison report to {csv_path} and {html_path}")
        
        return df
    
    def plot_learning_curves(self, experiments: Optional[List[str]] = None):
        """Plot learning curves for selected experiments"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        experiments = experiments or list(self.experiments.keys())
        
        for exp_name in experiments:
            if exp_name not in self.experiments:
                continue
            
            exp_data = self.experiments[exp_name]
            
            # Plot training loss
            if 'train_loss' in exp_data['metrics']:
                train_loss = exp_data['metrics']['train_loss']
                steps = range(len(train_loss))
                axes[0].plot(steps, train_loss, label=exp_name, alpha=0.8, linewidth=2)
            
            # Plot validation loss
            if 'val_loss' in exp_data['metrics']:
                val_loss = exp_data['metrics']['val_loss']
                steps = range(len(val_loss))
                axes[1].plot(steps, val_loss, label=exp_name, alpha=0.8, linewidth=2)
        
        axes[0].set_xlabel('Steps')
        axes[0].set_ylabel('Training Loss')
        axes[0].set_title('Training Loss Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Steps')
        axes[1].set_ylabel('Validation Loss')
        axes[1].set_title('Validation Loss Comparison')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.output_dir / 'learning_curves_comparison.png'
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"📊 Saved learning curves to {plot_path}")
    
    def plot_metrics_comparison(self):
        """Create comprehensive metrics comparison plots"""
        df = self.generate_comparison_report()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Experiment Metrics Comparison', fontsize=16)
        
        # Best validation loss
        ax = axes[0, 0]
        df_sorted = df.sort_values('best_val_loss')
        ax.barh(range(len(df_sorted)), df_sorted['best_val_loss'].values)
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted['experiment'].values)
        ax.set_xlabel('Best Validation Loss')
        ax.set_title('Best Validation Loss')
        ax.invert_yaxis()
        
        # Training time
        ax = axes[0, 1]
        df_sorted = df.sort_values('total_time_hours')
        ax.barh(range(len(df_sorted)), df_sorted['total_time_hours'].values)
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted['experiment'].values)
        ax.set_xlabel('Total Time (hours)')
        ax.set_title('Training Time')
        ax.invert_yaxis()
        
        # Peak memory usage
        ax = axes[0, 2]
        df_sorted = df.sort_values('peak_gpu_memory_gb')
        ax.barh(range(len(df_sorted)), df_sorted['peak_gpu_memory_gb'].values)
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted['experiment'].values)
        ax.set_xlabel('Peak GPU Memory (GB)')
        ax.set_title('Peak Memory Usage')
        ax.invert_yaxis()
        
        # Efficiency: Loss reduction per hour
        ax = axes[1, 0]
        df['efficiency'] = (df['min_train_loss'] - df['final_train_loss']) / df['total_time_hours']
        df_sorted = df.sort_values('efficiency')
        ax.barh(range(len(df_sorted)), df_sorted['efficiency'].values)
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted['experiment'].values)
        ax.set_xlabel('Loss Reduction per Hour')
        ax.set_title('Training Efficiency')
        ax.invert_yaxis()
        
        # Batch size vs learning rate scatter
        ax = axes[1, 1]
        if 'batch_size' in df.columns and 'learning_rate' in df.columns:
            scatter = ax.scatter(df['batch_size'], df['learning_rate'], 
                               c=df['best_val_loss'], s=100, cmap='viridis')
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Batch Size vs Learning Rate')
            ax.set_yscale('log')
            plt.colorbar(scatter, ax=ax, label='Best Val Loss')
            
            # Add experiment labels
            for idx, row in df.iterrows():
                ax.annotate(row['experiment'][:10], 
                          (row['batch_size'], row['learning_rate']),
                          fontsize=8, alpha=0.7)
        
        # Best epoch distribution
        ax = axes[1, 2]
        ax.hist(df['best_epoch'].values, bins=20, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Best Epoch')
        ax.set_ylabel('Count')
        ax.set_title('Best Epoch Distribution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.output_dir / 'metrics_comparison.png'
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"📊 Saved metrics comparison to {plot_path}")
    
    def find_best_experiment(self, metric: str = 'best_val_loss') -> Tuple[str, Dict]:
        """Find the best experiment based on a specific metric"""
        best_exp = None
        best_value = float('inf') if 'loss' in metric else -float('inf')
        
        for exp_name, exp_data in self.experiments.items():
            value = exp_data.get(metric)
            
            if value is not None:
                if 'loss' in metric:
                    if value < best_value:
                        best_value = value
                        best_exp = exp_name
                else:
                    if value > best_value:
                        best_value = value
                        best_exp = exp_name
        
        if best_exp:
            logger.info(f"🏆 Best experiment for {metric}: {best_exp} (value: {best_value})")
            return best_exp, self.experiments[best_exp]
        else:
            logger.warning(f"Could not find best experiment for metric: {metric}")
            return None, {}
    
    def create_latex_table(self, output_file: str = 'comparison_table.tex'):
        """Create LaTeX table for paper/report"""
        df = self.generate_comparison_report()
        
        # Select key columns for table
        columns = ['experiment', 'best_val_loss', 'final_train_loss', 
                  'total_time_hours', 'peak_gpu_memory_gb']
        
        df_selected = df[columns].round(3)
        
        # Convert to LaTeX
        latex_table = df_selected.to_latex(
            index=False,
            caption='Experiment Comparison Results',
            label='tab:experiment_comparison',
            column_format='l' + 'r' * (len(columns) - 1)
        )
        
        # Save to file
        latex_path = self.output_dir / output_file
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        
        logger.info(f"📊 Saved LaTeX table to {latex_path}")
        
        return latex_table
    
    def export_best_config(self, metric: str = 'best_val_loss') -> Dict:
        """Export configuration of the best experiment"""
        best_exp, best_data = self.find_best_experiment(metric)
        
        if best_exp and 'config' in best_data:
            config = best_data['config']
            
            # Save best config
            best_config_path = self.output_dir / 'best_config.json'
            with open(best_config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"📊 Exported best configuration to {best_config_path}")
            
            return config
        
        return {}