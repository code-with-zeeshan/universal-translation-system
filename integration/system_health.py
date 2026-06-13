# integration/system_health.py
"""
System health monitoring for the Universal Translation System
"""

import time
import psutil
import torch
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from prometheus_client import Gauge

logger = logging.getLogger(__name__)

system_health = Gauge('system_health_status', 'System health status (1=healthy, 0=unhealthy)')


class SystemHealthMonitor:
    """Monitor system health and performance"""

    def __init__(self, system: 'UniversalTranslationSystem'):
        self.system = system
        self.health_metrics = {}
        self.executor = ThreadPoolExecutor(max_workers=1)

    async def check_health(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        health = {
            'status': 'healthy',
            'timestamp': time.time(),
            'components': {},
            'performance': {},
            'resources': {}
        }

        # Check each component
        components = [
            ('data_pipeline', self._check_data_pipeline),
            ('vocab_manager', self._check_vocab_manager),
            ('encoder', self._check_encoder),
            ('decoder', self._check_decoder),
            ('trainer', self._check_trainer),
            ('evaluator', self._check_evaluator)
        ]

        for name, check_func in components:
            try:
                health['components'][name] = await check_func()
            except Exception as e:
                health['components'][name] = {'status': 'error', 'error': str(e)}
                health['status'] = 'degraded'

        # Check resources
        health['resources'] = await self._check_resources()

        # Update Prometheus metric
        system_health.set(1 if health['status'] == 'healthy' else 0)

        return health

    async def _check_data_pipeline(self) -> Dict[str, Any]:
        """Check data pipeline health"""
        return {
            'status': 'healthy' if self.system.data_pipeline is not None else 'not_initialized',
            'data_dir_exists': Path(self.system.config.data_dir).exists(),
            'processed_data_exists': (Path(self.system.config.data_dir) / "processed").exists()
        }

    async def _check_vocab_manager(self) -> Dict[str, Any]:
        """Check vocabulary manager health"""
        if self.system.vocab_manager is None:
            return {'status': 'not_initialized'}

        try:
            # Check loaded vocabularies
            loaded_versions = self.system.vocab_manager.get_loaded_versions()
            return {
                'status': 'healthy',
                'loaded_packs': len(loaded_versions),
                'versions': loaded_versions
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    async def _check_encoder(self) -> Dict[str, Any]:
        """Check encoder health"""
        if self.system.encoder is None:
            return {'status': 'not_initialized'}

        return {
            'status': 'healthy',
            'device': str(next(self.system.encoder.parameters()).device),
            'parameters': sum(p.numel() for p in self.system.encoder.parameters()),
            'training': self.system.encoder.training
        }

    async def _check_decoder(self) -> Dict[str, Any]:
        """Check decoder health"""
        if self.system.decoder is None:
            return {'status': 'not_initialized'}

        return {
            'status': 'healthy',
            'device': str(next(self.system.decoder.parameters()).device),
            'parameters': sum(p.numel() for p in self.system.decoder.parameters()),
            'training': self.system.decoder.training
        }

    async def _check_trainer(self) -> Dict[str, Any]:
        """Check trainer health"""
        return {
            'status': 'healthy' if self.system.trainer is not None else 'not_initialized'
        }

    async def _check_evaluator(self) -> Dict[str, Any]:
        """Check evaluator health"""
        return {
            'status': 'healthy' if self.system.evaluator is not None else 'not_initialized'
        }

    async def _check_resources(self) -> Dict[str, Any]:
        """Check system resources"""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        resources = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3)
        }

        # GPU resources
        if torch.cuda.is_available():
            resources['gpu'] = {
                'memory_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'memory_reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                'utilization': self._get_gpu_utilization()
            }

        return resources

    def _get_gpu_utilization(self) -> Optional[float]:
        """Get GPU utilization percentage"""
        try:
            import nvidia_ml_py3 as nvml
            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except Exception:
            return None

    def validate_configuration(self) -> List[str]:
        """Validate system configuration"""
        errors = []

        # Check directories exist
        for dir_attr in ['data_dir', 'model_dir', 'vocab_dir']:
            dir_path = Path(getattr(self.config, dir_attr))
            if not dir_path.exists():
                errors.append(f"{dir_attr} does not exist: {dir_path}")

        # Check device availability
        if self.config.device == 'cuda' and not torch.cuda.is_available():
            errors.append("CUDA requested but not available")

        # Check port availability for monitoring
        if self.config.enable_monitoring:
            import socket
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('localhost', self.config.monitoring_port))
                if result == 0:
                    errors.append(f"Monitoring port {self.config.monitoring_port} already in use")
                sock.close()
            except Exception:
                pass

        return errors
