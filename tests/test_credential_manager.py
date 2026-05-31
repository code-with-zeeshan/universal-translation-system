"""Tests for utils.credential_manager"""
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest
from utils.credential_manager import CredentialManager, get_credential, set_credential, delete_credential


class TestCredentialManager:
    @pytest.fixture
    def manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "creds.json"
            cm = CredentialManager(
                app_name="TestApp",
                env_prefix="TEST_",
                config_path=str(cfg_path),
                use_keyring=False,
            )
            yield cm

    def test_get_set_delete(self, manager):
        manager.set("api_key", "secret-123", store_in="file")
        val = manager.get("api_key")
        assert val == "secret-123"
        assert manager.delete("api_key") is True
        assert manager.get("api_key") is None

    def test_get_default(self, manager):
        val = manager.get("nonexistent", default="fallback")
        assert val == "fallback"

    def test_get_from_env(self, manager):
        os.environ["TEST_MYKEY"] = "env-value"
        try:
            val = manager.get("mykey")
            assert val == "env-value"
        finally:
            del os.environ["TEST_MYKEY"]

    def test_cache_hit(self, manager):
        manager.set("cached_key", "cached_value", store_in="file")
        # First call loads into cache
        assert manager.get("cached_key") == "cached_value"
        # Modify underlying file to verify cache is used
        with open(manager.config_path) as f:
            config = json.load(f)
        config["cached_key"] = "modified"
        with open(manager.config_path, "w") as f:
            json.dump(config, f)
        # Cache should still return original
        assert manager.get("cached_key") == "cached_value"

    def test_list_keys(self, manager):
        os.environ["TEST_KEY1"] = "val1"
        os.environ["TEST_KEY2"] = "val2"
        try:
            keys = manager.list_keys()
            assert "key1" in keys
            assert "key2" in keys
        finally:
            del os.environ["TEST_KEY1"]
            del os.environ["TEST_KEY2"]

    def test_clear(self, manager):
        manager.set("k1", "v1", store_in="file")
        manager.set("k2", "v2", store_in="file")
        manager.clear()
        assert manager.get("k1") is None
        assert manager.get("k2") is None

    def test_delete_from_cache_only(self, manager):
        manager.set("temp_key", "temp_value", store_in="file")
        assert manager.delete("temp_key") is True
        assert manager.get("temp_key") is None

    def test_delete_nonexistent(self, manager):
        assert manager.delete("no_such_key") is False

    def test_encrypted_storage(self, manager):
        """When encryption_key is provided, stored values should be encrypted."""
        cm = CredentialManager(
            app_name="EncTest",
            env_prefix="ENC_",
            use_keyring=False,
            encryption_key="test-encryption-key-32bytes!",
        )
        try:
            cm.set("secret", "my-secret-value", store_in="file")
            # Read the raw file - should contain encrypted text
            with open(cm.config_path) as f:
                raw = json.load(f)
            assert raw["secret"].startswith("encrypted:")
            # get() should decrypt
            assert cm.get("secret") == "my-secret-value"
        finally:
            if cm.config_path.exists():
                cm.config_path.unlink()
            cm.config_path.parent.rmdir()

    def test_global_functions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "global_creds.json"
            with patch("utils.credential_manager.credential_manager") as mock_cm:
                mock_cm.get.return_value = "global-val"
                assert get_credential("some_key") == "global-val"
                set_credential("k", "v", store_in="file")
                mock_cm.set.assert_called_with("k", "v", store_in="file")
                mock_cm.delete.return_value = True
                assert delete_credential("k") is True
