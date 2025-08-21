# utils/resource_monitor.py
"""
Resource monitoring utilities for the Universal Translation System
"""
import psutil
import time
import threading
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
import logging
from collections import defaultdict, deque
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class ResourceMonitor:
    """Monitor system resources during training and data processing"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history = defaultdict(lambda: deque(maxlen=max_history))
        self.active_monitors = {}
        self.lock = threading.Lock()
        self._monitoring = False
        self._monitor_thread = None
        
    def start_monitoring(self, interval: float = 1.0):
        """Start continuous resource monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"Started resource monitoring with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop continuous resource monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Stopped resource monitoring")
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                metrics = self._collect_metrics()
                timestamp = time.time()
                
                with self.lock:
                    for key, value in metrics.items():
                        self.metrics_history[key].append((timestamp, value))
                
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def _collect_metrics(self) -> Dict[str, float]:
        """Collect current system metrics"""
        metrics = {}
        
        # CPU metrics
        metrics['cpu_percent'] = psutil.cpu_percent()
        metrics['cpu_count'] = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics['memory_percent'] = memory.percent
        metrics['memory_used_gb'] = memory.used / (1024**3)
        metrics['memory_available_gb'] = memory.available / (1024**3)
        metrics['memory_total_gb'] = memory.total / (1024**3)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        metrics['disk_percent'] = (disk.used / disk.total) * 100
        metrics['disk_used_gb'] = disk.used / (1024**3)
        metrics['disk_free_gb'] = disk.free / (1024**3)
        
        # GPU metrics (if available)
        try:
            import torch
            if torch.cuda.is_available():
                metrics['gpu_memory_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
                metrics['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
                metrics['gpu_memory_total_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                metrics['gpu_utilization'] = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
        except ImportError:
            pass
        
        # Network I/O
        net_io = psutil.net_io_counters()
        if net_io:
            metrics['network_bytes_sent'] = net_io.bytes_sent
            metrics['network_bytes_recv'] = net_io.bytes_recv
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        if disk_io:
            metrics['disk_read_bytes'] = disk_io.read_bytes
            metrics['disk_write_bytes'] = disk_io.write_bytes
        
        return metrics
    
    @contextmanager
    def monitor(self, operation_name: str):
        """Context manager for monitoring specific operations"""
        start_time = time.time()
        start_metrics = self._collect_metrics()
        
        with self.lock:
            self.active_monitors[operation_name] = {
                'start_time': start_time,
                'start_metrics': start_metrics
            }
        
        try:
            yield
        finally:
            end_time = time.time()
            end_metrics = self._collect_metrics()
            duration = end_time - start_time
            
            # Calculate deltas
            deltas = {}
            for key in start_metrics:
                if key in end_metrics:
                    deltas[f"{key}_delta"] = end_metrics[key] - start_metrics[key]
            
            operation_stats = {
                'duration': duration,
                'start_metrics': start_metrics,
                'end_metrics': end_metrics,
                'deltas': deltas
            }
            
            with self.lock:
                self.active_monitors[operation_name] = operation_stats
            
            logger.info(f"Operation '{operation_name}' completed in {duration:.2f}s")
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        return self._collect_metrics()
    
    def get_metrics_history(self, metric_name: str, last_n: Optional[int] = None) -> List[tuple]:
        """Get history for a specific metric"""
        with self.lock:
            history = list(self.metrics_history[metric_name])
        
        if last_n:
            return history[-last_n:]
        return history
    
    def get_operation_stats(self, operation_name: str) -> Optional[Dict]:
        """Get statistics for a specific operation"""
        with self.lock:
            return self.active_monitors.get(operation_name)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive resource usage summary"""
        current_metrics = self._collect_metrics()
        
        summary = {
            'current_metrics': current_metrics,
            'active_operations': len(self.active_monitors),
            'monitoring_active': self._monitoring,
            'history_length': {
                metric: len(history) 
                for metric, history in self.metrics_history.items()
            }
        }
        
        # Add peak values from history
        peaks = {}
        with self.lock:
            for metric_name, history in self.metrics_history.items():
                if history:
                    values = [value for _, value in history]
                    peaks[f"{metric_name}_peak"] = max(values)
                    peaks[f"{metric_name}_min"] = min(values)
                    peaks[f"{metric_name}_avg"] = sum(values) / len(values)
        
        summary['peaks'] = peaks
        
        return summary
    
    def save_report(self, filepath: str):
        """Save monitoring report to file"""
        report = {
            'timestamp': time.time(),
            'summary': self.get_summary(),
            'operation_stats': dict(self.active_monitors),
            'metrics_history': {
                metric: list(history)[-100:]  # Last 100 entries
                for metric, history in self.metrics_history.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Resource monitoring report saved to {filepath}")
    
    def check_resource_limits(self, 
                            max_memory_percent: float = 90.0,
                            max_disk_percent: float = 95.0,
                            max_gpu_memory_percent: float = 95.0) -> Dict[str, bool]:
        """Check if resource usage is within acceptable limits"""
        metrics = self._collect_metrics()
        
        checks = {
            'memory_ok': metrics.get('memory_percent', 0) < max_memory_percent,
            'disk_ok': metrics.get('disk_percent', 0) < max_disk_percent,
            'gpu_memory_ok': True  # Default to OK if no GPU
        }
        
        # GPU memory check
        if 'gpu_memory_total_gb' in metrics and metrics['gpu_memory_total_gb'] > 0:
            gpu_usage_percent = (metrics.get('gpu_memory_allocated_gb', 0) / 
                               metrics['gpu_memory_total_gb']) * 100
            checks['gpu_memory_ok'] = gpu_usage_percent < max_gpu_memory_percent
        
        return checks
    
    def get_resource_recommendations(self) -> List[str]:
        """Get recommendations based on current resource usage"""
        metrics = self._collect_metrics()
        recommendations = []
        
        # Memory recommendations
        if metrics.get('memory_percent', 0) > 80:
            recommendations.append("High memory usage detected. Consider reducing batch size or enabling gradient checkpointing.")
        
        # GPU memory recommendations
        if 'gpu_memory_total_gb' in metrics:
            gpu_usage = (metrics.get('gpu_memory_allocated_gb', 0) / 
                        metrics['gpu_memory_total_gb']) * 100
            if gpu_usage > 85:
                recommendations.append("High GPU memory usage. Consider mixed precision training or smaller model.")
        
        # CPU recommendations
        if metrics.get('cpu_percent', 0) > 90:
            recommendations.append("High CPU usage. Consider reducing data loading workers or preprocessing complexity.")
        
        # Disk recommendations
        if metrics.get('disk_percent', 0) > 90:
            recommendations.append("Low disk space. Consider cleaning up temporary files or using external storage.")
        
        return recommendations
    
    def __del__(self):
        """Cleanup when monitor is destroyed"""
        self.stop_monitoring()

# Global resource monitor instance
resource_monitor = ResourceMonitor()

class ResourceAlert:
    """Alert system for resource monitoring"""
    
    def __init__(self, monitor: ResourceMonitor):
        self.monitor = monitor
        self.alert_thresholds = {
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'gpu_memory_percent': 90.0
        }
        self.alert_callbacks = []
    
    def add_alert_callback(self, callback):
        """Add callback function for alerts"""
        self.alert_callbacks.append(callback)
    
    def check_alerts(self):
        """Check for alert conditions"""
        metrics = self.monitor.get_current_metrics()
        alerts = []
        
        # Check memory
        if metrics.get('memory_percent', 0) > self.alert_thresholds['memory_percent']:
            alerts.append({
                'type': 'memory',
                'level': 'warning',
                'message': f"High memory usage: {metrics['memory_percent']:.1f}%",
                'value': metrics['memory_percent']
            })
        
        # Check disk
        if metrics.get('disk_percent', 0) > self.alert_thresholds['disk_percent']:
            alerts.append({
                'type': 'disk',
                'level': 'warning',
                'message': f"Low disk space: {metrics['disk_percent']:.1f}% used",
                'value': metrics['disk_percent']
            })
        
        # Check GPU memory
        if 'gpu_memory_total_gb' in metrics and metrics['gpu_memory_total_gb'] > 0:
            gpu_usage = (metrics.get('gpu_memory_allocated_gb', 0) / 
                        metrics['gpu_memory_total_gb']) * 100
            if gpu_usage > self.alert_thresholds['gpu_memory_percent']:
                alerts.append({
                    'type': 'gpu_memory',
                    'level': 'warning',
                    'message': f"High GPU memory usage: {gpu_usage:.1f}%",
                    'value': gpu_usage
                })
        
        # Trigger callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
        
        return alerts

def log_alert(alert: Dict):
    """Default alert callback that logs alerts"""
    logger.warning(f"RESOURCE ALERT: {alert['message']}")

# Create default alert system
resource_alert = ResourceAlert(resource_monitor)
resource_alert.add_alert_callback(log_alert)