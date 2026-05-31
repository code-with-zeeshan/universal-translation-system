# training/quantization_common.py
"""
Common quantization utilities for the Universal Translation System
"""

import torch
import logging
from dataclasses import dataclass
from utils.exceptions import TrainingError

logger = logging.getLogger(__name__)


def fake_quantize_tensor(tensor: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    """Fake quantization with straight-through estimator for QAT."""
    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1

    min_val = tensor.min()
    max_val = tensor.max()

    scale = (max_val - min_val) / (qmax - qmin)
    scale = torch.clamp(scale, min=1e-8)
    zero_point = qmin - torch.round(min_val / scale)
    zero_point = torch.clamp(zero_point, qmin, qmax)

    tensor_q = torch.round(tensor / scale + zero_point)
    tensor_q = torch.clamp(tensor_q, qmin, qmax)
    tensor_dq = (tensor_q - zero_point) * scale

    return tensor + (tensor_dq - tensor).detach()


@dataclass
class QuantizationConfig:
    """Configuration for quantization"""
    calibration_samples: int = 1000
    backend: str = 'fbgemm'  # 'fbgemm' for x86, 'qnnpack' for ARM
    symmetric_quantization: bool = True
    per_channel_quantization: bool = True
    preserve_embeddings: bool = True  # Keep embeddings in higher precision

    def __post_init__(self):
        """Validate configuration"""
        if self.backend not in ['fbgemm', 'qnnpack']:
            raise TrainingError(f"Invalid backend: {self.backend}")
        if self.calibration_samples < 100:
            logger.warning("Low calibration samples may result in poor quantization")
        if self.backend == 'qnnpack' and not self._is_arm():
            logger.warning("qnnpack backend is optimized for ARM, using on x86")

    def _is_arm(self) -> bool:
        import platform
        return 'arm' in platform.machine().lower()


@dataclass
class QualityMetrics:
    """Metrics for model quality comparison"""
    latency_ms: float
    memory_mb: float
    bleu_score: float
    accuracy: float
    perplexity: float
    compression_ratio: float
