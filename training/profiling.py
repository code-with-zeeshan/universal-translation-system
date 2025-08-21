# training/profiling.py
"""
Profiling utilities for training optimization.
Extracted from train_universal_system.py
"""
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
import numpy as np
import pandas as pd
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class ProfileGuidedTrainer:
    """Training with profiling for optimization insights"""
    
    def __init__(self, base_trainer):
        """
        Args:
            base_trainer: Any trainer instance (e.g., IntelligentTrainer)
        """
        self.base_trainer = base_trainer
        self.profiling_results = {}
        self.optimization_suggestions = []
    
    def profile_training_step(self, num_steps: int = 10, 
                              trace_path: str = "./profiler_traces") -> Dict[str, Any]:
        """Profile training to find bottlenecks"""
        
        # Create dataloader for profiling
        train_loader = self.base_trainer._create_train_loader() if hasattr(self.base_trainer, '_create_train_loader') else \
                      self.base_trainer.memory_trainer.create_optimized_dataloader(
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
                if hasattr(self.base_trainer, '_training_step'):
                    loss = self.base_trainer._training_step(batch)
                else:
                    # Fallback for other trainer types
                    loss = self._run_generic_training_step(batch)
                
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
        self._print_profiling_summary(results)
        
        return results
    
    def _run_generic_training_step(self, batch):
        """Fallback training step for profiling"""
        # Prepare batch
        batch = self.base_trainer._prepare_batch(batch) if hasattr(self.base_trainer, '_prepare_batch') else batch
        
        # Compute loss
        loss = self.base_trainer._compute_loss(batch) if hasattr(self.base_trainer, '_compute_loss') else \
               self.base_trainer._forward_pass(**batch)
        
        # Backward
        loss.backward()
        
        # Optimizer step
        self.base_trainer.optimizer.step()
        self.base_trainer.optimizer.zero_grad()
        
        return loss.item()
    
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
        if hasattr(self.base_trainer, 'strategy'):
            if not self.base_trainer.strategy.memory_config.compile_model:
                suggestions.append("Enable torch.compile for potential speedup")
        
        return list(set(suggestions))  # Remove duplicates
    
    def _save_profiling_report(self, results: Dict, trace_path: str):
        """Save detailed profiling report"""
        report_path = Path(trace_path) / "profiling_report.json"
        report_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Prepare serializable report
        report = {
            'timestamp': time.time(),
            'average_step_time': float(np.mean(results['step_times'])),
            'step_times': results['step_times'],
            'memory_usage': results['memory_usage'],
            'bottlenecks': results['bottlenecks'],
            'suggestions': self.optimization_suggestions
        }
        
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Saved profiling report to {report_path}")
    
    def _print_profiling_summary(self, results: Dict):
        """Print profiling summary"""
        print("\n" + "="*60)
        print("PROFILING SUMMARY")
        print("="*60)
        print(f"Average step time: {np.mean(results['step_times']):.3f}s")
        
        if results['memory_usage']:
            avg_memory = np.mean([m['allocated'] for m in results['memory_usage']])
            print(f"Average GPU memory: {avg_memory:.2f}GB")
        
        print(f"\nTop time-consuming operations:")
        print(results['top_operations'])
        
        print("\nOptimization suggestions:")
        for suggestion in self.optimization_suggestions:
            print(f"  - {suggestion}")
        print("="*60)
    
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
            
            train_loader = self.base_trainer._create_train_loader()
            
            for i, batch in enumerate(train_loader):
                if i >= num_steps:
                    break
                
                start_time = time.time()
                start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
                # Training step
                loss = self.base_trainer._training_step(batch)
                
                # Record metrics
                step_times.append(time.time() - start_time)
                if torch.cuda.is_available():
                    memory_usage.append((torch.cuda.memory_allocated() - start_memory) / 1024**3)
            
            # Record results
            results.append({
                'config_name': config.get('name', 'unnamed'),
                'batch_size': config.get('batch_size', self.base_trainer.batch_sizer.current_batch_size),
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
            'compile_model': getattr(self.base_trainer.strategy.memory_config, 'compile_model', False),
            'mixed_precision': getattr(self.base_trainer.strategy.memory_config, 'mixed_precision', False),
            'gradient_checkpointing': getattr(self.base_trainer.strategy.memory_config, 'gradient_checkpointing', False)
        }
    
    def _apply_config(self, config: Dict):
        """Apply configuration changes"""
        if 'batch_size' in config:
            self.base_trainer.batch_sizer.current_batch_size = config['batch_size']
        if 'mixed_precision' in config:
            self.base_trainer.strategy.memory_config.mixed_precision = config['mixed_precision']
        # Add more config applications as needed
    
    def _restore_config(self, config: Dict):
        """Restore saved configuration"""
        self._apply_config(config)