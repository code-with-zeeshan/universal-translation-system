# universal-decoder-node/universal_decoder_node/utils/profiler.py
import time
import logging
import threading
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Union
from contextlib import contextmanager
from collections import defaultdict, deque
import functools

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class FunctionProfiler:
    """
    Profiler for tracking function execution time and resource usage.
    
    Features:
    - Function execution time tracking
    - Call count statistics
    - Memory usage tracking
    - Bottleneck identification
    - Export to various formats
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'FunctionProfiler':
        """Get singleton instance of FunctionProfiler"""
        if cls._instance is None:
            cls._instance = FunctionProfiler()
        return cls._instance
    
    def __init__(self, config=None, max_history: int = 1000):
        self.max_history = max_history
        
        # Try to import config if not provided
        if config is None:
            try:
                from ..config import DecoderConfig, ProfilingConfig
                self.config = ProfilingConfig()
            except ImportError:
                # Fallback to environment variables
                self.config = None
        else:
            self.config = config
            
        # Set parameters from config or environment variables
        if self.config:
            self.enabled = self.config.enable_profiling
            self.profile_dir = self.config.profile_output_dir
            self.bottleneck_threshold_ms = self.config.bottleneck_threshold_ms
            self.export_format = self.config.export_format
        else:
            self.enabled = os.environ.get("ENABLE_PROFILING", "false").lower() == "true"
            self.profile_dir = os.environ.get("PROFILE_OUTPUT_DIR", "profiles")
            self.bottleneck_threshold_ms = float(os.environ.get("BOTTLENECK_THRESHOLD_MS", "100.0"))
            self.export_format = os.environ.get("PROFILE_EXPORT_FORMAT", "json")
        
        # Create profile directory if it doesn't exist
        if self.enabled:
            Path(self.profile_dir).mkdir(parents=True, exist_ok=True)
        
        # Stats storage
        self.stats = defaultdict(lambda: {
            'call_count': 0,
            'total_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'avg_time': 0.0,
            'last_time': 0.0,
            'memory_delta': 0.0,
            'history': deque(maxlen=max_history)
        })
        
        self.lock = threading.RLock()
        
        logger.info(f"Function profiler {'enabled' if self.enabled else 'disabled'}")
    
    def profile(self, func=None, *, name: Optional[str] = None):
        """
        Decorator for profiling functions.
        
        Args:
            func: Function to profile
            name: Custom name for the function in profiling data
            
        Returns:
            Decorated function
        """
        def decorator(f):
            if not self.enabled:
                return f
                
            func_name = name or f.__qualname__
            
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                with self.profile_context(func_name):
                    return f(*args, **kwargs)
            return wrapper
            
        if func is None:
            return decorator
        return decorator(func)
    
    @contextmanager
    def profile_context(self, name: str):
        """
        Context manager for profiling code blocks.
        
        Args:
            name: Name for the code block in profiling data
        """
        if not self.enabled:
            yield
            return
            
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory if end_memory is not None and start_memory is not None else 0
            
            self._record_stats(name, duration, memory_delta)
    
    def _record_stats(self, name: str, duration: float, memory_delta: float):
        """Record profiling statistics"""
        with self.lock:
            stats = self.stats[name]
            stats['call_count'] += 1
            stats['total_time'] += duration
            stats['min_time'] = min(stats['min_time'], duration)
            stats['max_time'] = max(stats['max_time'], duration)
            stats['avg_time'] = stats['total_time'] / stats['call_count']
            stats['last_time'] = duration
            stats['memory_delta'] = memory_delta
            
            # Record history with timestamp
            stats['history'].append({
                'timestamp': time.time(),
                'duration': duration,
                'memory_delta': memory_delta
            })
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in GB"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024**3)
        except ImportError:
            return None
    
    def get_stats(self, name: Optional[str] = None) -> Dict:
        """
        Get profiling statistics.
        
        Args:
            name: Function name to get stats for, or None for all stats
            
        Returns:
            Dictionary of profiling statistics
        """
        with self.lock:
            if name:
                return dict(self.stats[name])
            
            return {k: dict(v) for k, v in self.stats.items()}
    
    def get_sorted_stats(self, sort_by: str = 'total_time', limit: int = 10) -> List[Dict]:
        """
        Get sorted profiling statistics.
        
        Args:
            sort_by: Field to sort by ('total_time', 'avg_time', 'call_count', 'max_time')
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries with profiling statistics
        """
        with self.lock:
            stats_list = []
            
            for name, stats in self.stats.items():
                # Skip entries with no calls
                if stats['call_count'] == 0:
                    continue
                    
                stats_dict = dict(stats)
                stats_dict['name'] = name
                stats_list.append(stats_dict)
            
            # Sort by the specified field
            stats_list.sort(key=lambda x: x[sort_by], reverse=True)
            
            # Limit results
            return stats_list[:limit]
    
    def reset_stats(self, name: Optional[str] = None):
        """
        Reset profiling statistics.
        
        Args:
            name: Function name to reset stats for, or None for all stats
        """
        with self.lock:
            if name:
                if name in self.stats:
                    self.stats[name] = {
                        'call_count': 0,
                        'total_time': 0.0,
                        'min_time': float('inf'),
                        'max_time': 0.0,
                        'avg_time': 0.0,
                        'last_time': 0.0,
                        'memory_delta': 0.0,
                        'history': deque(maxlen=self.max_history)
                    }
            else:
                self.stats.clear()
    
    def export_stats(self, filepath: Optional[str] = None, format: str = None) -> Optional[str]:
        """
        Export profiling statistics to a file.
        
        Args:
            filepath: Path to export to, or None to use default
            format: Export format ('json', 'csv', 'txt'). If None, uses the configured format.
            
        Returns:
            Path to the exported file
        """
        if not self.enabled:
            return None
            
        # Use provided format or fall back to configured format
        if format is None:
            format = self.export_format
            
        # Use default filepath if none provided
        if filepath is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filepath = os.path.join(self.profile_dir, f"profile-{timestamp}.{format}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Get stats
        stats = self.get_sorted_stats(limit=1000)
        
        # Export based on format
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump({
                    'timestamp': time.time(),
                    'stats': stats
                }, f, indent=2, default=str)
        elif format == 'csv':
            import csv
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['name', 'call_count', 'total_time', 'avg_time', 'min_time', 'max_time', 'memory_delta'])
                
                for stat in stats:
                    writer.writerow([
                        stat['name'],
                        stat['call_count'],
                        stat['total_time'],
                        stat['avg_time'],
                        stat['min_time'],
                        stat['max_time'],
                        stat['memory_delta']
                    ])
        elif format == 'txt':
            with open(filepath, 'w') as f:
                f.write("FUNCTION PROFILING REPORT\n")
                f.write("========================\n\n")
                f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("Top functions by total time:\n")
                f.write("---------------------------\n")
                for stat in self.get_sorted_stats(sort_by='total_time', limit=10):
                    f.write(f"{stat['name']}: {stat['total_time']:.6f}s ({stat['call_count']} calls, avg: {stat['avg_time']:.6f}s)\n")
                
                f.write("\nTop functions by average time:\n")
                f.write("-----------------------------\n")
                for stat in self.get_sorted_stats(sort_by='avg_time', limit=10):
                    f.write(f"{stat['name']}: {stat['avg_time']:.6f}s ({stat['call_count']} calls, total: {stat['total_time']:.6f}s)\n")
                
                f.write("\nTop functions by call count:\n")
                f.write("---------------------------\n")
                for stat in self.get_sorted_stats(sort_by='call_count', limit=10):
                    f.write(f"{stat['name']}: {stat['call_count']} calls (total: {stat['total_time']:.6f}s, avg: {stat['avg_time']:.6f}s)\n")
        else:
            logger.error(f"Unsupported export format: {format}")
            return None
        
        logger.info(f"Exported profiling stats to {filepath}")
        return filepath
    
    def print_stats(self, sort_by: str = 'total_time', limit: int = 10):
        """
        Print profiling statistics to the console.
        
        Args:
            sort_by: Field to sort by ('total_time', 'avg_time', 'call_count', 'max_time')
            limit: Maximum number of results to print
        """
        if not self.enabled:
            print("Profiling is disabled")
            return
            
        stats = self.get_sorted_stats(sort_by=sort_by, limit=limit)
        
        print("\n" + "="*80)
        print(f"FUNCTION PROFILING REPORT (sorted by {sort_by})")
        print("="*80)
        
        if not stats:
            print("No profiling data available")
            print("="*80)
            return
        
        # Print header
        print(f"{'Function':<50} {'Calls':<8} {'Total (s)':<12} {'Avg (s)':<12} {'Min (s)':<12} {'Max (s)':<12}")
        print("-"*80)
        
        # Print stats
        for stat in stats:
            print(f"{stat['name']:<50} {stat['call_count']:<8} {stat['total_time']:<12.6f} "
                  f"{stat['avg_time']:<12.6f} {stat['min_time']:<12.6f} {stat['max_time']:<12.6f}")
        
        print("="*80)
    
    def identify_bottlenecks(self, threshold_ms: float = None) -> List[Dict]:
        """
        Identify potential bottlenecks in the code.
        
        Args:
            threshold_ms: Threshold in milliseconds for identifying bottlenecks.
                          If None, uses the configured threshold.
            
        Returns:
            List of potential bottlenecks
        """
        # Use provided threshold or fall back to configured threshold
        if threshold_ms is None:
            threshold_ms = self.bottleneck_threshold_ms
            
        bottlenecks = []
        
        for name, stats in self.get_stats().items():
            # Skip entries with few calls
            if stats['call_count'] < 5:
                continue
                
            # Check if average time exceeds threshold
            if stats['avg_time'] * 1000 > threshold_ms:
                bottlenecks.append({
                    'name': name,
                    'avg_time_ms': stats['avg_time'] * 1000,
                    'call_count': stats['call_count'],
                    'total_time': stats['total_time'],
                    'severity': 'high' if stats['avg_time'] * 1000 > threshold_ms * 2 else 'medium'
                })
        
        # Sort by average time
        bottlenecks.sort(key=lambda x: x['avg_time_ms'], reverse=True)
        
        return bottlenecks

# Create global instance
function_profiler = FunctionProfiler.get_instance()

# Decorator for easy profiling
def profile(func=None, *, name: Optional[str] = None):
    """
    Decorator for profiling functions.
    
    Args:
        func: Function to profile
        name: Custom name for the function in profiling data
        
    Returns:
        Decorated function
    """
    return function_profiler.profile(func, name=name)

@contextmanager
def profile_section(name: str):
    """
    Context manager for profiling code sections.
    
    Args:
        name: Name for the code section in profiling data
    """
    with function_profiler.profile_context(name):
        yield