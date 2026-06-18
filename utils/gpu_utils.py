# utils/gpu_utils.py
"""GPU profile detection + per-tier optimal config for all pipeline stages."""
import logging
import os
from dataclasses import dataclass
from typing import Optional

try:
    import torch
except ImportError:
    torch = None  # type: ignore

logger = logging.getLogger(__name__)

_gpu_optimized = False


# ── GPU Tier Definitions ────────────────────────────────────────────────

@dataclass
class GPUProfile:
    """Optimal batch sizes, worker counts, and feature flags per GPU tier."""
    name: str
    memory_gb: float
    # download
    download_max_workers: int = 16
    download_rate_limit: int = 10
    download_parallel_batches: bool = True
    # NLLB inference (augment + distill)
    nllb_batch_size: int = 128
    nllb_flash_attention_2: bool = True
    nllb_torch_compile: bool = False
    nllb_bettertransformer: bool = True
    nllb_dtype: str = "float16"
    nllb_multi_gpu: bool = False
    # COMET
    comet_batch_size: int = 64
    # sample_filter (LaBSE)
    labse_batch_size: int = 64
    # create_ready
    create_ready_workers: int = 4
    # vocabulary
    vocab_threads: int = 8


GPU_TIERS: dict[str, GPUProfile] = {
    "t4": GPUProfile(
        name="t4", memory_gb=16,
        download_max_workers=8, download_rate_limit=8,
        nllb_batch_size=128, nllb_flash_attention_2=True,
        nllb_bettertransformer=True, nllb_dtype="float16",
        comet_batch_size=64, labse_batch_size=64,
        create_ready_workers=4, vocab_threads=8,
    ),
    "l4": GPUProfile(
        name="l4", memory_gb=24,
        download_max_workers=12, download_rate_limit=10,
        nllb_batch_size=256, nllb_flash_attention_2=True,
        nllb_torch_compile=True, nllb_bettertransformer=True, nllb_dtype="float16",
        comet_batch_size=128, labse_batch_size=128,
        create_ready_workers=8, vocab_threads=12,
    ),
    "l40s": GPUProfile(
        name="l40s", memory_gb=48,
        download_max_workers=24, download_rate_limit=15,
        nllb_batch_size=512, nllb_flash_attention_2=True,
        nllb_torch_compile=True, nllb_bettertransformer=True, nllb_dtype="float16",
        comet_batch_size=256, labse_batch_size=256,
        create_ready_workers=12, vocab_threads=16,
    ),
    "a100": GPUProfile(
        name="a100", memory_gb=80,
        download_max_workers=32, download_rate_limit=20,
        nllb_batch_size=1024, nllb_flash_attention_2=True,
        nllb_torch_compile=True, nllb_bettertransformer=True,
        nllb_dtype="bfloat16", nllb_multi_gpu=True,
        comet_batch_size=512, labse_batch_size=512,
        create_ready_workers=16, vocab_threads=32,
    ),
    "h100": GPUProfile(
        name="h100", memory_gb=80,
        download_max_workers=48, download_rate_limit=30,
        nllb_batch_size=2048, nllb_flash_attention_2=True,
        nllb_torch_compile=True, nllb_bettertransformer=True,
        nllb_dtype="bfloat16", nllb_multi_gpu=True,
        comet_batch_size=1024, labse_batch_size=1024,
        create_ready_workers=24, vocab_threads=48,
    ),
    "cpu": GPUProfile(
        name="cpu", memory_gb=0,
        download_max_workers=6, download_rate_limit=4,
        nllb_batch_size=1, nllb_flash_attention_2=False,
        nllb_bettertransformer=True, nllb_dtype="float32",
        comet_batch_size=1, labse_batch_size=1,
        create_ready_workers=4, vocab_threads=8,
    ),
}


def detect_gpu_tier() -> str:
    """Detect GPU hardware and return matching tier name."""
    if torch is None or not torch.cuda.is_available():
        return "cpu"
    try:
        props = torch.cuda.get_device_properties(0)
        gb = props.total_memory / 1e9
        name = props.name.lower()
        if "h100" in name or "h200" in name:
            return "h100"
        if "a100" in name or "a10" in name:
            return "a100"
        if "l40" in name or "l40s" in name:
            return "l40s"
        if "l4" in name or "l16" in name or "t4" in name:
            # Distinguish L4 (24GB) from T4 (16GB)
            return "l4" if gb >= 22 else "t4"
        if "v100" in name:
            return "l4"  # V100 (16-32GB) close to L4 tier
        # Fallback by memory
        if gb >= 70:
            return "a100"
        if gb >= 40:
            return "l40s"
        if gb >= 20:
            return "l4"
        return "t4"
    except Exception:
        return "cpu"


def get_gpu_profile(override: Optional[str] = None) -> GPUProfile:
    """Return GPUProfile for detected (or overridden) tier."""
    tier = override or os.environ.get("UTS_GPU_TIER") or detect_gpu_tier()
    profile = GPU_TIERS.get(tier)
    if profile is None:
        logger.warning(f"Unknown GPU tier '{tier}', falling back to T4")
        profile = GPU_TIERS["t4"]
    logger.info(f"📊 GPU profile: {profile.name} ({profile.memory_gb:.0f}GB)")
    return profile


# ── Legacy helpers ──────────────────────────────────────────────────────

def optimize_gpu_memory():
    """Optimize GPU memory settings (idempotent, guarded against DataLoader workers)."""
    global _gpu_optimized
    if _gpu_optimized:
        return
    if torch is None:
        return
    import torch.utils.data
    if torch.utils.data.get_worker_info() is not None:
        return
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.95)
        torch.cuda.empty_cache()
        os.environ.setdefault(
            'PYTORCH_CUDA_ALLOC_CONF',
            'max_split_size_mb:512,expandable_segments:True,roundup_power2_divisions:16'
        )
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        _gpu_optimized = True
        logger.info("✅ GPU memory optimized")


def get_gpu_memory_info():
    """Get current GPU memory usage."""
    if torch is not None and torch.cuda.is_available():
        return {
            'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
            'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
            'free_gb': (torch.cuda.get_device_properties(0).total_memory -
                       torch.cuda.memory_reserved()) / 1024**3,
        }
    return {}


def torch_is_available() -> bool:
    """Check if torch is available and CUDA is accessible."""
    if torch is None:
        return False
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def configure_nllb_model(model, profile: GPUProfile):
    """Apply NLLB inference optimizations based on GPU profile."""
    if torch is None:
        return model
    if profile.nllb_bettertransformer:
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)
            logger.info("  ✅ BetterTransformer enabled")
        except Exception:
            logger.debug("BetterTransformer not available, skipping")
    if profile.nllb_torch_compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            logger.info("  ✅ torch.compile enabled")
        except Exception:
            logger.debug("torch.compile failed, skipping")
    return model