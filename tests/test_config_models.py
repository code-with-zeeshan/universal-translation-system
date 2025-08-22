"""
Tests for the configuration models.
"""

import os
import pytest
import tempfile
from pathlib import Path
import yaml
import json

from config.config_models import (
    EncoderConfig,
    DecoderConfig,
    CoordinatorConfig,
    CircuitBreakerConfig,
    MonitoringConfig,
    TrainingConfig,
    SystemConfig,
    load_config
)


class TestEncoderConfig:
    """Tests for EncoderConfig."""
    
    def test_default_values(self):
        """Test default values."""
        config = EncoderConfig()
        assert config.model_path == "models/production/encoder.pt"
        assert config.embedding_dim == 768
        assert config.vocab_dir == "vocabs"
        assert config.max_sequence_length == 512
        assert config.device == "auto"
        assert config.fallback_model_path is None
    
    def test_environment_variables(self, monkeypatch):
        """Test environment variable overrides."""
        monkeypatch.setenv("ENCODER_MODEL_PATH", "custom/path/model.pt")
        monkeypatch.setenv("EMBEDDING_DIM", "1024")
        monkeypatch.setenv("VOCAB_DIR", "custom/vocabs")
        monkeypatch.setenv("MAX_SEQUENCE_LENGTH", "256")
        monkeypatch.setenv("ENCODER_DEVICE", "cpu")
        monkeypatch.setenv("FALLBACK_MODEL_PATH", "custom/path/fallback.pt")
        
        config = EncoderConfig()
        assert config.model_path == "custom/path/model.pt"
        assert config.embedding_dim == 1024
        assert config.vocab_dir == "custom/vocabs"
        assert config.max_sequence_length == 256
        assert config.device == "cpu"
        assert config.fallback_model_path == "custom/path/fallback.pt"
    
    def test_device_validation(self):
        """Test device validation."""
        # Valid values
        EncoderConfig(device="auto")
        EncoderConfig(device="cpu")
        EncoderConfig(device="cuda")
        
        # Invalid value
        with pytest.raises(ValueError):
            EncoderConfig(device="invalid")


class TestDecoderConfig:
    """Tests for DecoderConfig."""
    
    def test_default_values(self):
        """Test default values."""
        config = DecoderConfig()
        assert config.model_path == "models/production/decoder.pt"
        assert config.vocab_dir == "vocabs"
        assert config.max_sequence_length == 512
        assert config.device == "cuda"
        assert config.batch_size == 32
        assert config.beam_size == 5
        assert config.max_batch_tokens == 8192
    
    def test_environment_variables(self, monkeypatch):
        """Test environment variable overrides."""
        monkeypatch.setenv("DECODER_MODEL_PATH", "custom/path/decoder.pt")
        monkeypatch.setenv("VOCAB_DIR", "custom/vocabs")
        monkeypatch.setenv("MAX_SEQUENCE_LENGTH", "256")
        monkeypatch.setenv("DECODER_DEVICE", "cpu")
        monkeypatch.setenv("DECODER_BATCH_SIZE", "64")
        monkeypatch.setenv("DECODER_BEAM_SIZE", "10")
        monkeypatch.setenv("MAX_BATCH_TOKENS", "16384")
        
        config = DecoderConfig()
        assert config.model_path == "custom/path/decoder.pt"
        assert config.vocab_dir == "custom/vocabs"
        assert config.max_sequence_length == 256
        assert config.device == "cpu"
        assert config.batch_size == 64
        assert config.beam_size == 10
        assert config.max_batch_tokens == 16384
    
    def test_device_validation(self):
        """Test device validation."""
        # Valid values
        DecoderConfig(device="cpu")
        DecoderConfig(device="cuda")
        
        # Invalid value
        with pytest.raises(ValueError):
            DecoderConfig(device="invalid")


class TestCoordinatorConfig:
    """Tests for CoordinatorConfig."""
    
    def test_default_values(self):
        """Test default values."""
        # This will raise an error because jwt_secret is required
        with pytest.raises(ValueError):
            CoordinatorConfig()
    
    def test_with_jwt_secret(self):
        """Test with JWT secret."""
        config = CoordinatorConfig(jwt_secret="a" * 32)
        assert config.host == "0.0.0.0"
        assert config.port == 8002
        assert config.workers == 1
        assert config.title == "Universal Translation Coordinator"
        assert config.decoder_pool == ["decoder:8001"]
        assert config.jwt_secret == "a" * 32
        assert config.token_expiry == 3600
    
    def test_environment_variables(self, monkeypatch):
        """Test environment variable overrides."""
        monkeypatch.setenv("COORDINATOR_HOST", "127.0.0.1")
        monkeypatch.setenv("COORDINATOR_PORT", "5000")
        monkeypatch.setenv("COORDINATOR_WORKERS", "4")
        monkeypatch.setenv("COORDINATOR_TITLE", "Custom Coordinator")
        monkeypatch.setenv("DECODER_POOL", "decoder1:8001,decoder2:8001")
        monkeypatch.setenv("COORDINATOR_JWT_SECRET", "a" * 32)
        monkeypatch.setenv("TOKEN_EXPIRY", "7200")
        
        config = CoordinatorConfig()
        assert config.host == "127.0.0.1"
        assert config.port == 5000
        assert config.workers == 4
        assert config.title == "Custom Coordinator"
        assert config.decoder_pool == ["decoder1:8001", "decoder2:8001"]
        assert config.jwt_secret == "a" * 32
        assert config.token_expiry == 7200
    
    def test_jwt_secret_validation(self):
        """Test JWT secret validation."""
        # Valid value
        CoordinatorConfig(jwt_secret="a" * 32)
        
        # Empty value
        with pytest.raises(ValueError):
            CoordinatorConfig(jwt_secret="")
        
        # Too short
        with pytest.raises(ValueError):
            CoordinatorConfig(jwt_secret="short")


class TestSystemConfig:
    """Tests for SystemConfig."""
    
    def test_default_values(self):
        """Test default values."""
        # This will raise an error because jwt_secret is required
        with pytest.raises(ValueError):
            SystemConfig()
    
    def test_with_jwt_secret(self):
        """Test with JWT secret."""
        config = SystemConfig(coordinator=CoordinatorConfig(jwt_secret="a" * 32))
        assert isinstance(config.encoder, EncoderConfig)
        assert isinstance(config.decoder, DecoderConfig)
        assert isinstance(config.coordinator, CoordinatorConfig)
        assert isinstance(config.circuit_breaker, CircuitBreakerConfig)
        assert isinstance(config.monitoring, MonitoringConfig)
        assert isinstance(config.training, TrainingConfig)
    
    def test_from_yaml(self):
        """Test loading from YAML."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            yaml_path = f.name
            yaml.dump({
                "coordinator": {
                    "jwt_secret": "a" * 32
                }
            }, f)
        
        try:
            config = SystemConfig.from_yaml(yaml_path)
            assert config.coordinator.jwt_secret == "a" * 32
        finally:
            os.unlink(yaml_path)
    
    def test_from_json(self):
        """Test loading from JSON."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = f.name
            json.dump({
                "coordinator": {
                    "jwt_secret": "a" * 32
                }
            }, f)
        
        try:
            config = SystemConfig.from_json(json_path)
            assert config.coordinator.jwt_secret == "a" * 32
        finally:
            os.unlink(json_path)
    
    def test_to_yaml(self):
        """Test saving to YAML."""
        config = SystemConfig(coordinator=CoordinatorConfig(jwt_secret="a" * 32))
        
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            yaml_path = f.name
        
        try:
            config.to_yaml(yaml_path)
            
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
            
            assert data["coordinator"]["jwt_secret"] == "a" * 32
        finally:
            os.unlink(yaml_path)
    
    def test_to_json(self):
        """Test saving to JSON."""
        config = SystemConfig(coordinator=CoordinatorConfig(jwt_secret="a" * 32))
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = f.name
        
        try:
            config.to_json(json_path)
            
            with open(json_path, "r") as f:
                data = json.load(f)
            
            assert data["coordinator"]["jwt_secret"] == "a" * 32
        finally:
            os.unlink(json_path)


class TestLoadConfig:
    """Tests for load_config function."""
    
    def test_load_from_yaml(self):
        """Test loading from YAML."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            yaml_path = f.name
            yaml.dump({
                "coordinator": {
                    "jwt_secret": "a" * 32
                }
            }, f)
        
        try:
            config = load_config(yaml_path)
            assert config.coordinator.jwt_secret == "a" * 32
        finally:
            os.unlink(yaml_path)
    
    def test_load_from_json(self):
        """Test loading from JSON."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = f.name
            json.dump({
                "coordinator": {
                    "jwt_secret": "a" * 32
                }
            }, f)
        
        try:
            config = load_config(json_path)
            assert config.coordinator.jwt_secret == "a" * 32
        finally:
            os.unlink(json_path)
    
    def test_load_from_environment(self, monkeypatch):
        """Test loading from environment variables."""
        monkeypatch.setenv("COORDINATOR_JWT_SECRET", "a" * 32)
        
        # This will still raise an error because we're not mocking the validator
        with pytest.raises(ValueError):
            load_config()
    
    def test_file_not_found(self):
        """Test file not found error."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")
    
    def test_unsupported_format(self):
        """Test unsupported format error."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            txt_path = f.name
        
        try:
            with pytest.raises(ValueError):
                load_config(txt_path)
        finally:
            os.unlink(txt_path)