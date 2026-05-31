"""Tests for utils.artifact_store"""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from utils.artifact_store import ArtifactStore, StoreConfig


class TestStoreConfig:
    def test_store_config_dataclass(self):
        cfg = StoreConfig(
            repo_id="test/repo",
            token=None,
            revision="main",
            models_dir=Path("/tmp/models"),
            vocabs_dir=Path("/tmp/vocabs"),
            adapters_dir=Path("/tmp/adapters"),
        )
        assert cfg.repo_id == "test/repo"
        assert cfg.revision == "main"
        assert cfg.models_dir == Path("/tmp/models")


class TestArtifactStore:
    @pytest.fixture
    def cfg(self):
        return StoreConfig(
            repo_id="test/repo",
            token="test-token",
            revision="main",
            models_dir=Path("/tmp/test-models"),
            vocabs_dir=Path("/tmp/test-vocabs"),
            adapters_dir=Path("/tmp/test-adapters"),
        )

    def test_init_with_config(self, cfg):
        store = ArtifactStore(config=cfg)
        assert store.cfg.repo_id == "test/repo"

    def test_init_without_config_raises(self):
        """Without HF_HUB_REPO_ID env var, init should raise."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="HF_HUB_REPO_ID"):
                ArtifactStore()

    def test_init_without_config_from_env(self):
        with patch.dict(os.environ, {
            "HF_HUB_REPO_ID": "env/repo",
            "HF_TOKEN": "env-token",
            "HF_HUB_REVISION": "dev",
        }):
            store = ArtifactStore()
            assert store.cfg.repo_id == "env/repo"
            assert store.cfg.token == "env-token"
            assert store.cfg.revision == "dev"

    def test_ensure_model_already_exists(self, cfg):
        store = ArtifactStore(config=cfg)
        local = cfg.models_dir / "production" / "encoder.onnx"
        local.parent.mkdir(parents=True, exist_ok=True)
        local.touch()
        result = store.ensure_model("production/encoder.onnx")
        assert result == local

    def test_ensure_model_downloads(self, cfg):
        store = ArtifactStore(config=cfg)
        repo_path = "models/production/encoder.onnx"
        local = cfg.models_dir / "production" / "encoder.onnx"
        with patch.object(store, "_download_file", return_value=local) as mock_dl:
            result = store.ensure_model("production/encoder.onnx")
            mock_dl.assert_called_once_with(repo_path, local)
            assert result == local

    def test_ensure_model_without_models_prefix(self, cfg):
        store = ArtifactStore(config=cfg)
        local = cfg.models_dir / "encoder.onnx"
        with patch.object(store, "_download_file", return_value=local) as mock_dl:
            result = store.ensure_model("encoder.onnx")
            mock_dl.assert_called_once_with("models/encoder.onnx", local)

    def test_ensure_adapter_already_exists(self, cfg):
        store = ArtifactStore(config=cfg)
        local = cfg.adapters_dir / "es.bin"
        local.parent.mkdir(parents=True, exist_ok=True)
        local.touch()
        result = store.ensure_adapter("es")
        assert result == local

    def test_ensure_adapter_downloads(self, cfg):
        store = ArtifactStore(config=cfg)
        repo_path = "models/adapters/es.bin"
        local = cfg.adapters_dir / "es.bin"
        with patch.object(store, "_download_file", return_value=local) as mock_dl:
            result = store.ensure_adapter("es")
            mock_dl.assert_called_once_with(repo_path, local)

    def test_ensure_vocab_pack_with_version(self, cfg):
        store = ArtifactStore(config=cfg)
        local = cfg.vocabs_dir / "latin_v1.msgpack"
        with patch.object(store, "_download_file", return_value=local) as mock_dl:
            result = store.ensure_vocab_pack("latin", version="1")
            mock_dl.assert_called_once_with("vocabs/latin_v1.msgpack", local)

    def test_ensure_vocab_pack_version_already_exists(self, cfg):
        store = ArtifactStore(config=cfg)
        local = cfg.vocabs_dir / "latin_v2.msgpack"
        local.parent.mkdir(parents=True, exist_ok=True)
        local.touch()
        result = store.ensure_vocab_pack("latin", version="2")
        assert result == local

    def test_download_file_raises_if_hf_unavailable(self, cfg):
        store = ArtifactStore(config=cfg)
        with patch("utils.artifact_store.hf_hub_download", None):
            with pytest.raises(RuntimeError, match="huggingface_hub is not available"):
                store._download_file("test/path", Path("/tmp/test"))

    def test_ensure_for_language_pair(self, cfg):
        store = ArtifactStore(config=cfg)
        with patch.object(store, "ensure_vocab_pack") as mock_vocab:
            with patch.object(store, "ensure_adapter") as mock_adapter:
                store.ensure_for_language_pair("en", "zh", adapter="zh-adapter")
                mock_vocab.assert_called_once_with("cjk")
                mock_adapter.assert_called_once_with("zh-adapter")

    def test_ensure_for_language_pair_no_adapter(self, cfg):
        store = ArtifactStore(config=cfg)
        with patch.object(store, "ensure_vocab_pack") as mock_vocab:
            store.ensure_for_language_pair("en", "fr")
            mock_vocab.assert_called_once_with("latin")
