import torch
import time
import logging

logger = logging.getLogger(__name__)


class MemoryTracker:
    """Advanced memory tracking and profiling"""
    
    def __init__(self):
        self.start_time = None
        self.step_times = []
        self.memory_usage = []
        self.peak_memory = 0
        
    def start_step(self):
        """Start tracking a training step"""
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def end_step(self):
        """End tracking a training step"""
        if self.start_time:
            step_time = time.time() - self.start_time
            self.step_times.append(step_time)
            
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
                self.memory_usage.append(current_memory)
                self.peak_memory = max(self.peak_memory, peak_memory)
    
    def log_memory_usage(self):
        """Log current memory usage"""
        if torch.cuda.is_available():
            current = torch.cuda.memory_allocated() / 1024**3
            peak = torch.cuda.max_memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            
            logger.info(f"📊 Memory - Current: {current:.2f}GB, Peak: {peak:.2f}GB, Reserved: {reserved:.2f}GB")
    
    def start_profiling(self):
        """Start comprehensive profiling"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.start_time = time.time()
    
    def stop_profiling(self):
        """Stop profiling and generate report"""
        if self.start_time:
            total_time = time.time() - self.start_time
            
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / 1024**3
                
                logger.info(f"📈 Profile Report:")
                logger.info(f"   Total time: {total_time:.2f}s")
                logger.info(f"   Peak memory: {peak_memory:.2f}GB")
                logger.info(f"   Average step time: {sum(self.step_times)/len(self.step_times):.4f}s")
