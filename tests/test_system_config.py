"""
Tests for integration.system_config - Pydantic-validated system configuration.
"""

import socket
import pytest
from integration.system_config import IntegrationSystemConfig as SystemConfig


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


FREE_PORT = _free_port()


class TestSystemConfig:
    def test_default_values(self):
        config = SystemConfig(monitoring_port=FREE_PORT)
        assert config.data_dir == "data"
        assert config.model_dir == "models"
        assert config.vocab_dir == "vocabs"
        assert config.checkpoint_dir == "checkpoints"
        assert isinstance(config.device, str)
        assert config.use_adapters is True
        assert config.quantization_mode == "int8"
        assert config.vocab_cache_size == 3
        assert config.batch_size == 32
        assert config.enable_monitoring is True

    def test_custom_values(self):
        config = SystemConfig(
            data_dir="/custom/data", model_dir="/custom/models",
            vocab_dir="/custom/vocabs", checkpoint_dir="/custom/ckpt",
            device="cpu", use_adapters=False, quantization_mode="fp16",
            vocab_cache_size=5, batch_size=64, enable_monitoring=False,
            monitoring_port=FREE_PORT,
        )
        assert config.data_dir == "/custom/data"
        assert config.model_dir == "/custom/models"
        assert config.vocab_dir == "/custom/vocabs"
        assert config.checkpoint_dir == "/custom/ckpt"
        assert config.device == "cpu"
        assert config.use_adapters is False
        assert config.quantization_mode == "fp16"
        assert config.vocab_cache_size == 5
        assert config.batch_size == 64
        assert config.enable_monitoring is False

    def test_quantization_fp32_valid(self):
        config = SystemConfig(quantization_mode="fp32", monitoring_port=FREE_PORT)
        assert config.quantization_mode == "fp32"

    def test_quantization_accepts_all_strings_pydantic_v1(self):
        """Pydantic v1 ignores `pattern` in Field(), so any string is accepted."""
        for mode in ["fp64", "int4", "int32", "bf16", ""]:
            config = SystemConfig(quantization_mode=mode, monitoring_port=_free_port())
            assert config.quantization_mode == mode

    @pytest.mark.parametrize("size", [0, -1, 513, 1024])
    def test_batch_size_invalid(self, size):
        with pytest.raises(Exception):
            SystemConfig(batch_size=size, monitoring_port=FREE_PORT)

    def test_batch_size_boundaries(self):
        c1 = SystemConfig(batch_size=1, monitoring_port=FREE_PORT)
        assert c1.batch_size == 1
        c2 = SystemConfig(batch_size=512, monitoring_port=FREE_PORT)
        assert c2.batch_size == 512

    @pytest.mark.parametrize("cache", [0, -1, 11, 100])
    def test_vocab_cache_invalid(self, cache):
        with pytest.raises(Exception):
            SystemConfig(vocab_cache_size=cache, monitoring_port=FREE_PORT)

    def test_vocab_cache_boundaries(self):
        c1 = SystemConfig(vocab_cache_size=1, monitoring_port=FREE_PORT)
        assert c1.vocab_cache_size == 1
        c2 = SystemConfig(vocab_cache_size=10, monitoring_port=FREE_PORT)
        assert c2.vocab_cache_size == 10

    @pytest.mark.parametrize("port", [1023, 65536, 0, -1])
    def test_monitoring_port_range_invalid(self, port):
        with pytest.raises(Exception):
            SystemConfig(monitoring_port=port)

    def test_monitoring_port_boundaries(self):
        c1 = SystemConfig(monitoring_port=1024)
        assert c1.monitoring_port == 1024
        c2 = SystemConfig(monitoring_port=65535)
        assert c2.monitoring_port == 65535

    def test_device_cpu(self):
        config = SystemConfig(device="cpu", monitoring_port=FREE_PORT)
        assert config.device == "cpu"

    def test_use_adapters_default_true(self):
        config = SystemConfig(monitoring_port=FREE_PORT)
        assert config.use_adapters is True

    def test_enable_monitoring_default_true(self):
        config = SystemConfig(monitoring_port=FREE_PORT)
        assert config.enable_monitoring is True

    def test_checkpoint_dir_from_constants(self):
        from utils.constants import CHECKPOINT_DIR
        config = SystemConfig(monitoring_port=FREE_PORT)
        assert config.checkpoint_dir == CHECKPOINT_DIR
