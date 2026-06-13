# integration/system_config.py
"""
System configuration with Pydantic validation
"""

import logging
import torch
from pydantic import BaseModel, Field, validator
from utils.common_utils import RuntimeDirectoryManager

logger = logging.getLogger(__name__)


class IntegrationSystemConfig(BaseModel):
    """Configuration for the integrated system with validation"""
    data_dir: str = Field(default_factory=lambda: str(RuntimeDirectoryManager().data_dir), description="Data directory path")
    model_dir: str = Field(default_factory=lambda: str(RuntimeDirectoryManager().models_dir), description="Model directory path")
    vocab_dir: str = Field(default_factory=lambda: str(RuntimeDirectoryManager().vocab_dir), description="Vocabulary directory path")
    checkpoint_dir: str = Field(default_factory=lambda: str(RuntimeDirectoryManager().checkpoints_dir), description="Checkpoint directory")
    device: str = Field(default="cuda", description="Device for computation")
    use_adapters: bool = Field(default=True, description="Use language adapters")
    quantization_mode: str = Field(default="int8", pattern="^(fp32|fp16|int8)$")
    vocab_cache_size: int = Field(default=3, ge=1, le=10)
    batch_size: int = Field(default=32, ge=1, le=512)
    enable_monitoring: bool = Field(default=True, description="Enable Prometheus monitoring")
    monitoring_port: int = Field(default=8000, ge=1024, le=65535)

    @validator('device')
    def validate_device(cls, v):
        if v == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return 'cpu'
        return v

    @validator('batch_size')
    def validate_batch_size(cls, v, values):
        if 'device' in values and values['device'] == 'cpu' and v > 32:
            logger.warning(f"Large batch size ({v}) on CPU may be slow")
        return v

    @validator('vocab_cache_size')
    def validate_vocab_cache_size(cls, v):
        import psutil
        available_memory_gb = psutil.virtual_memory().available / 1024**3
        estimated_cache_size_gb = v * 0.5  # Assume ~500MB per vocab pack

        if estimated_cache_size_gb > available_memory_gb * 0.5:
            logger.warning(f"Vocab cache size may use {estimated_cache_size_gb:.1f}GB RAM")
        return v

    @validator('monitoring_port')
    def validate_monitoring_port(cls, v):
        import socket
        try:
            # Check if port is available
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', v))
            return v
        except OSError:
            raise ValueError(f"Port {v} is already in use")

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
