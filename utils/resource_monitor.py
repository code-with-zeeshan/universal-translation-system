# utils/resource_monitor.py
import psutil
import torch
import logging
from contextlib import contextmanager
from typing import Dict, Any
import time

logger = logging.getLogger(__name__)

class ResourceMonitor:
    def __init__(self):
        self.history = []
        
    @contextmanager
    def monitor(self, label: str):
        """Monitor resource usage for a code block"""
        # Start measurements
        start_time = time.time()
        start_cpu = psutil.cpu_percent(interval=0.1)
        start_memory = psutil.Process().memory_info().rss / 1024**3  # GB
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_gpu_memory = torch.cuda.memory_allocated() / 1024**3
        else:
            start_gpu_memory = 0
        
        try:
            yield
        finally:
            # End measurements
            end_time = time.time()
            end_cpu = psutil.cpu_percent(interval=0.1)
            end_memory = psutil.Process().memory_info().rss / 1024**3
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                end_gpu_memory = torch.cuda.memory_allocated() / 1024**3
            else:
                end_gpu_memory = 0
            
            # Calculate differences
            metrics = {
                'label': label,
                'duration': end_time - start_time,
                'cpu_percent': (start_cpu + end_cpu) / 2,
                'memory_delta_gb': end_memory - start_memory,
                'gpu_memory_delta_gb': end_gpu_memory - start_gpu_memory,
                'timestamp': time.time()
            }
            
            self.history.append(metrics)
            
            # Log
            logger.info(f"Resource usage for '{label}':")
            logger.info(f"  Duration: {metrics['duration']:.2f}s")
            logger.info(f"  CPU: {metrics['cpu_percent']:.1f}%")
            logger.info(f"  Memory Δ: {metrics['memory_delta_gb']:.2f}GB")
            if torch.cuda.is_available():
                logger.info(f"  GPU Memory Δ: {metrics['gpu_memory_delta_gb']:.2f}GB")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all monitored operations"""
        if not self.history:
            return {}
        
        total_duration = sum(m['duration'] for m in self.history)
        avg_cpu = sum(m['cpu_percent'] for m in self.history) / len(self.history)
        total_memory_delta = sum(m['memory_delta_gb'] for m in self.history)
        total_gpu_memory_delta = sum(m['gpu_memory_delta_gb'] for m in self.history)
        
        return {
            'total_duration': total_duration,
            'average_cpu_percent': avg_cpu,
            'total_memory_delta_gb': total_memory_delta,
            'total_gpu_memory_delta_gb': total_gpu_memory_delta,
            'operation_count': len(self.history)
        }

# Global instance
resource_monitor = ResourceMonitor()