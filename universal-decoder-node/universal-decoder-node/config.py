# universal-decoder-node/universal_decoder_node/config.py
from dataclasses import dataclass
from typing import Optional
import yaml
import os


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
    jwt_secret: str = 'jwtsecret123'
    enable_auth: bool = True
    prometheus_port: int = 9200
    enable_tracing: bool = False


def load_config(config_path: str) -> DecoderConfig:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    decoder_cfg = data.get('decoder', {})
    security_cfg = data.get('security', {})
    monitoring_cfg = data.get('monitoring', {})
    
    return DecoderConfig(
        host=decoder_cfg.get('host', '0.0.0.0'),
        port=decoder_cfg.get('port', 8000),
        workers=decoder_cfg.get('workers', 1),
        model_path=decoder_cfg.get('model_path'),
        vocab_dir=decoder_cfg.get('vocab_dir', 'vocabs'),
        device=decoder_cfg.get('device', 'cuda'),
        max_batch_size=decoder_cfg.get('max_batch_size', 64),
        batch_timeout_ms=decoder_cfg.get('batch_timeout_ms', 10),
        jwt_secret=security_cfg.get('jwt_secret', 'jwtsecret123'),
        enable_auth=security_cfg.get('enable_auth', True),
        prometheus_port=monitoring_cfg.get('prometheus_port', 9200),
        enable_tracing=monitoring_cfg.get('enable_tracing', False)
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
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)