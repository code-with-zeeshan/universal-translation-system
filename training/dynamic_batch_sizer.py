import torch
import logging
import time

logger = logging.getLogger(__name__)


class DynamicBatchSizer:
    """Dynamic batch sizing based on memory usage"""
    
    def __init__(self, initial_batch_size: int = 32, max_batch_size: int = 128):
        self.current_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = 8
        self.memory_threshold = 0.9  # 90% of GPU memory
        self.adjustment_factor = 1.2
        
    def adjust_batch_size(self) -> int:
        """Adjust batch size based on memory usage"""
        if not torch.cuda.is_available():
            return self.current_batch_size
        
        memory_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        
        if memory_usage > self.memory_threshold:
            # Reduce batch size
            new_size = max(self.min_batch_size, int(self.current_batch_size / self.adjustment_factor))
            if new_size != self.current_batch_size:
                logger.info(f"🔽 Reducing batch size: {self.current_batch_size} -> {new_size}")
                self.current_batch_size = new_size
        elif memory_usage < 0.7:
            # Increase batch size
            new_size = min(self.max_batch_size, int(self.current_batch_size * self.adjustment_factor))
            if new_size != self.current_batch_size:
                logger.info(f"🔼 Increasing batch size: {self.current_batch_size} -> {new_size}")
                self.current_batch_size = new_size
        
        return self.current_batch_size
