# utils/gpu_utils.py
import torch
import gc
import logging

logger = logging.getLogger(__name__)

# Setup comprehensive memory optimizations
def optimize_gpu_memory():
    """Optimize GPU memory settings"""
    if torch.cuda.is_available():
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Set allocator settings
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True,roundup_power2_divisions:16'
        
        # Enable TF32 for better performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        logger.info("✅ GPU memory optimized")

def get_gpu_memory_info():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        return {
            'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
            'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
            'free_gb': (torch.cuda.get_device_properties(0).total_memory - 
                       torch.cuda.memory_reserved()) / 1024**3
        }
    return {}