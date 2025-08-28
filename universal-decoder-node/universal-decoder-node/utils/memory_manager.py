# universal-decoder-node/universal_decoder_node/utils/memory_manager.py
import gc
import logging
import threading
import time
import os
from typing import Dict, List, Optional, Callable, Any
import torch

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Memory management utility for the universal-decoder-node.
    
    Features:
    - Periodic memory cleanup
    - Memory usage monitoring
    - Automatic cache clearing
    - GPU memory optimization
    - Memory usage alerts
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'MemoryManager':
        """Get singleton instance of MemoryManager"""
        if cls._instance is None:
            cls._instance = MemoryManager()
        return cls._instance
    
    def __init__(self, config=None):
        # Try to import config if not provided
        if config is None:
            try:
                from ..config import DecoderConfig, MemoryConfig
                self.config = MemoryConfig()
            except ImportError:
                # Fallback to environment variables
                self.config = None
                
        else:
            self.config = config
            
        # Set parameters from config or environment variables
        if self.config:
            self.monitoring_interval = self.config.monitoring_interval_seconds
            self.memory_threshold = self.config.memory_threshold_percent
            self.gpu_memory_threshold = self.config.gpu_memory_threshold_percent
            self.enable_monitoring = self.config.enable_monitoring
            self.auto_cleanup = self.config.auto_cleanup
            self.cleanup_threshold = self.config.cleanup_threshold_percent
        else:
            self.monitoring_interval = int(os.environ.get("MEMORY_MONITOR_INTERVAL_SECONDS", "60"))
            self.memory_threshold = float(os.environ.get("MEMORY_THRESHOLD_PERCENT", "85"))
            self.gpu_memory_threshold = float(os.environ.get("GPU_MEMORY_THRESHOLD_PERCENT", "85"))
            self.enable_monitoring = os.environ.get("ENABLE_MEMORY_MONITORING", "true").lower() == "true"
            self.auto_cleanup = os.environ.get("AUTO_MEMORY_CLEANUP", "true").lower() == "true"
            self.cleanup_threshold = float(os.environ.get("CLEANUP_THRESHOLD_PERCENT", "80"))
        
        self._monitoring = False
        self._monitor_thread = None
        self._stop_event = threading.Event()
        self._callbacks = []
        
        # Cache for memory stats
        self._memory_stats = {}
        self._last_cleanup_time = 0
        
        # Start monitoring if enabled
        if self.enable_monitoring:
            self.start_monitoring()
    
    def start_monitoring(self):
        """Start memory monitoring thread"""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="MemoryMonitorThread"
        )
        self._monitor_thread.start()
        logger.info(f"Started memory monitoring with {self.monitoring_interval}s interval")
    
    def stop_monitoring(self):
        """Stop memory monitoring thread"""
        if not self._monitoring:
            return
            
        self._monitoring = False
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Stopped memory monitoring")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self._stop_event.is_set():
            try:
                # Get memory stats
                stats = self.get_memory_stats()
                
                # Check thresholds
                self._check_thresholds(stats)
                
                # Perform cleanup if needed
                if stats.get('system_memory_percent', 0) > self.memory_threshold * 0.9 or \
                   stats.get('gpu_memory_percent', 0) > self.gpu_memory_threshold * 0.9:
                    self.cleanup()
                
                # Wait for next interval
                self._stop_event.wait(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")
                self._stop_event.wait(self.monitoring_interval)
    
    def _check_thresholds(self, stats: Dict[str, Any]):
        """Check memory thresholds and trigger callbacks if exceeded"""
        alerts = []
        
        # Check system memory
        if stats.get('system_memory_percent', 0) > self.memory_threshold:
            alerts.append({
                'type': 'system_memory',
                'level': 'warning',
                'message': f"High system memory usage: {stats['system_memory_percent']:.1f}%",
                'value': stats['system_memory_percent']
            })
        
        # Check GPU memory
        if stats.get('gpu_memory_percent', 0) > self.gpu_memory_threshold:
            alerts.append({
                'type': 'gpu_memory',
                'level': 'warning',
                'message': f"High GPU memory usage: {stats['gpu_memory_percent']:.1f}%",
                'value': stats['gpu_memory_percent']
            })
        
        # Trigger callbacks
        for alert in alerts:
            for callback in self._callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in memory alert callback: {e}")
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for memory alerts"""
        self._callbacks.append(callback)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics"""
        stats = {}
        
        # System memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            stats['system_memory_total_gb'] = memory.total / (1024**3)
            stats['system_memory_used_gb'] = memory.used / (1024**3)
            stats['system_memory_available_gb'] = memory.available / (1024**3)
            stats['system_memory_percent'] = memory.percent
        except ImportError:
            logger.warning("psutil not available, system memory stats will be limited")
            stats['system_memory_percent'] = 0
        
        # GPU memory
        if torch.cuda.is_available():
            try:
                device = torch.cuda.current_device()
                stats['gpu_memory_allocated_gb'] = torch.cuda.memory_allocated(device) / (1024**3)
                stats['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved(device) / (1024**3)
                stats['gpu_memory_total_gb'] = torch.cuda.get_device_properties(device).total_memory / (1024**3)
                stats['gpu_memory_percent'] = (stats['gpu_memory_allocated_gb'] / stats['gpu_memory_total_gb']) * 100
                
                # Get per-tensor memory usage if available
                if hasattr(torch.cuda, 'memory_stats'):
                    memory_stats = torch.cuda.memory_stats(device)
                    stats['gpu_memory_active_gb'] = memory_stats.get('active.all.current', 0) / (1024**3)
                    stats['gpu_memory_inactive_gb'] = memory_stats.get('inactive.all.current', 0) / (1024**3)
            except Exception as e:
                logger.error(f"Error getting GPU memory stats: {e}")
                stats['gpu_memory_percent'] = 0
        else:
            stats['gpu_memory_percent'] = 0
        
        # Cache stats
        self._memory_stats = stats
        
        return stats
    
    def cleanup(self, force: bool = False):
        """
        Perform memory cleanup.
        
        Args:
            force: Force cleanup even if last cleanup was recent
        """
        current_time = time.time()
        
        # Skip if last cleanup was too recent (unless forced)
        if not force and current_time - self._last_cleanup_time < 30:
            return
            
        logger.info("Performing memory cleanup")
        
        # Python garbage collection
        collected = gc.collect(generation=2)
        logger.info(f"Garbage collection: collected {collected} objects")
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            try:
                before = torch.cuda.memory_allocated() / (1024**3)
                torch.cuda.empty_cache()
                after = torch.cuda.memory_allocated() / (1024**3)
                logger.info(f"CUDA cache cleared: {before:.2f}GB -> {after:.2f}GB")
            except Exception as e:
                logger.error(f"Error clearing CUDA cache: {e}")
        
        self._last_cleanup_time = current_time
    
    @staticmethod
    def optimize_for_inference(model: torch.nn.Module) -> torch.nn.Module:
        """
        Optimize model for inference to reduce memory usage.
        
        Args:
            model: PyTorch model
            
        Returns:
            Optimized model
        """
        # Set to eval mode
        model.eval()
        
        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False
        
        # Try to optimize with torch.compile if available
        if hasattr(torch, 'compile') and torch.__version__ >= '2.0.0':
            try:
                logger.info("Optimizing model with torch.compile")
                model = torch.compile(model, mode='reduce-overhead')
            except Exception as e:
                logger.warning(f"Failed to optimize with torch.compile: {e}")
        
        return model
    
    @staticmethod
    def get_model_size(model: torch.nn.Module) -> float:
        """
        Get model size in GB.
        
        Args:
            model: PyTorch model
            
        Returns:
            Model size in GB
        """
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size = param_size + buffer_size
        return total_size / (1024**3)
    
    def __del__(self):
        """Cleanup when manager is destroyed"""
        self.stop_monitoring()

# Default log callback
def log_memory_alert(alert: Dict[str, Any]):
    """Default callback for memory alerts"""
    logger.warning(f"MEMORY ALERT: {alert['message']}")

# Create global instance
memory_manager = MemoryManager.get_instance()
memory_manager.add_alert_callback(log_memory_alert)