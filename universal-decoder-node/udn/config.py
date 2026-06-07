# universal-decoder-node/universal_decoder_node/config.py
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import yaml
import os


@dataclass
class ModelConfig:
    """Model configuration"""
    encoder_dim: int = 1024
    decoder_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    vocab_size: int = 50000
    max_length: int = 256
    dropout: float = 0.1

@dataclass
class MemoryConfig:
    """Memory management configuration"""
    enable_monitoring: bool = True
    monitoring_interval_seconds: int = 60
    memory_threshold_percent: float = 85
    gpu_memory_threshold_percent: float = 85
    auto_cleanup: bool = True
    cleanup_threshold_percent: float = 80

@dataclass
class ProfilingConfig:
    """Profiling configuration"""
    enable_profiling: bool = False
    profile_output_dir: str = "profiles"
    bottleneck_threshold_ms: float = 100.0
    export_format: str = "json"

@dataclass
class DecoderConfig:
    """Decoder configuration"""
    host: str = '0.0.0.0'
    port: int = 8000
    workers: int = 1
    model_path: Optional[str] = None
    vocab_dir: str = 'vocabs'
    device: str = 'cuda'
    max_batch_size: int = 64
    batch_timeout_ms: int = 10
    jwt_secret: str = ''
    enable_auth: bool = True
    prometheus_port: int = 9200
    enable_tracing: bool = False
    redis_url: Optional[str] = None
    enforce_https: bool = True
    https_port: int = 443
    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)


def load_config(config_path: str) -> DecoderConfig:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    decoder_cfg = data.get('decoder', {})
    security_cfg = data.get('security', {})
    monitoring_cfg = data.get('monitoring', {})
    model_cfg = data.get('model', {})
    redis_cfg = data.get('redis', {})
    memory_cfg = data.get('memory', {})
    profiling_cfg = data.get('profiling', {})
    https_cfg = data.get('https', {})
    
    # Create model config
    model_config = ModelConfig(
        encoder_dim=model_cfg.get('encoder_dim', 1024),
        decoder_dim=model_cfg.get('decoder_dim', 512),
        num_layers=model_cfg.get('num_layers', 6),
        num_heads=model_cfg.get('num_heads', 8),
        vocab_size=model_cfg.get('vocab_size', 50000),
        max_length=model_cfg.get('max_length', 256),
        dropout=model_cfg.get('dropout', 0.1)
    )
    
    # Create memory config
    memory_config = MemoryConfig(
        enable_monitoring=memory_cfg.get('enable_monitoring', True),
        monitoring_interval_seconds=memory_cfg.get('monitoring_interval_seconds', 60),
        memory_threshold_percent=memory_cfg.get('memory_threshold_percent', 85),
        gpu_memory_threshold_percent=memory_cfg.get('gpu_memory_threshold_percent', 85),
        auto_cleanup=memory_cfg.get('auto_cleanup', True),
        cleanup_threshold_percent=memory_cfg.get('cleanup_threshold_percent', 80)
    )
    
    # Create profiling config
    profiling_config = ProfilingConfig(
        enable_profiling=profiling_cfg.get('enable_profiling', False),
        profile_output_dir=profiling_cfg.get('profile_output_dir', 'profiles'),
        bottleneck_threshold_ms=profiling_cfg.get('bottleneck_threshold_ms', 100.0),
        export_format=profiling_cfg.get('export_format', 'json')
    )
    
    return DecoderConfig(
        host=decoder_cfg.get('host', '0.0.0.0'),
        port=decoder_cfg.get('port', 8000),
        workers=decoder_cfg.get('workers', 1),
        model_path=decoder_cfg.get('model_path'),
        vocab_dir=decoder_cfg.get('vocab_dir', 'vocabs'),
        device=decoder_cfg.get('device', 'cuda'),
        max_batch_size=decoder_cfg.get('max_batch_size', 64),
        batch_timeout_ms=decoder_cfg.get('batch_timeout_ms', 10),
        jwt_secret=security_cfg.get('jwt_secret', ''),
        enable_auth=security_cfg.get('enable_auth', True),
        prometheus_port=monitoring_cfg.get('prometheus_port', 9200),
        enable_tracing=monitoring_cfg.get('enable_tracing', False),
        redis_url=redis_cfg.get('url'),
        enforce_https=https_cfg.get('enforce', True),
        https_port=https_cfg.get('port', 443),
        model=model_config,
        memory=memory_config,
        profiling=profiling_config
    )


def save_config(config: DecoderConfig, config_path: str):
    """Save configuration to YAML file"""
    data = {
        'decoder': {
            'host': config.host,
            'port': config.port,
            'workers': config.workers,
            'model_path': config.model_path,
            'vocab_dir': config.vocab_dir,
            'device': config.device,
            'max_batch_size': config.max_batch_size,
            'batch_timeout_ms': config.batch_timeout_ms
        },
        'security': {
            'jwt_secret': config.jwt_secret,
            'enable_auth': config.enable_auth
        },
        'monitoring': {
            'prometheus_port': config.prometheus_port,
            'enable_tracing': config.enable_tracing
        },
        'https': {
            'enforce': config.enforce_https,
            'port': config.https_port
        },
        'memory': {
            'enable_monitoring': config.memory.enable_monitoring,
            'monitoring_interval_seconds': config.memory.monitoring_interval_seconds,
            'memory_threshold_percent': config.memory.memory_threshold_percent,
            'gpu_memory_threshold_percent': config.memory.gpu_memory_threshold_percent,
            'auto_cleanup': config.memory.auto_cleanup,
            'cleanup_threshold_percent': config.memory.cleanup_threshold_percent
        },
        'profiling': {
            'enable_profiling': config.profiling.enable_profiling,
            'profile_output_dir': config.profiling.profile_output_dir,
            'bottleneck_threshold_ms': config.profiling.bottleneck_threshold_ms,
            'export_format': config.profiling.export_format
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)