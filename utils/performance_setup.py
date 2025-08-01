# performance_setup.py
import torch
import os

def setup_performance_optimizations():
    """Enable all performance optimizations"""
    
    # PyTorch optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set thread settings
    torch.set_num_threads(os.cpu_count())
    
    # Memory allocator settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
    
    # Disable debug mode
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(enabled=False)
    torch.autograd.profiler.emit_nvtx(enabled=False)
    
    # Set to eval mode for inference
    torch.set_grad_enabled(False)  # Only for inference
    
    print("✅ Performance optimizations enabled")

# Call at start of script:
setup_performance_optimizations()