# utils/credential_manager.py
"""
Credential management using system keyring and encrypted file storage.
Delegates to keyring (primary) and cryptography (encrypted file fallback).
"""

import os
import json
import base64
import logging
import threading
from typing import Dict, Optional
from pathlib import Path

try:
    import keyring
except Exception:
    keyring = None

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
except Exception:
    Fernet = None

from .exceptions import SecurityError

logger = logging.getLogger(__name__)


class CredentialManager:
    def __init__(self, app_name: str = "UniversalTranslationSystem", env_prefix: str = "UTS_",
                 config_path: Optional[str] = None, use_keyring: bool = True,
                 encryption_key: Optional[str] = None):
        self.app_name = app_name
        self.env_prefix = env_prefix
        self.use_keyring = use_keyring
        self.config_path = Path(config_path or Path.home() / f".{app_name.lower()}" / "credentials.json")
        self._cache: Dict[str, str] = {}
        self._lock = threading.Lock()
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self._fernet: Optional[Fernet] = None
        if encryption_key and Fernet:
            salt = b'UniversalTranslationSystem'
            kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000)
            self._fernet = Fernet(base64.urlsafe_b64encode(kdf.derive(encryption_key.encode())))

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        env_key = f"{self.env_prefix}{key.upper()}"
        val = os.environ.get(env_key)
        if val:
            with self._lock:
                self._cache[key] = val
            return val
        if self.use_keyring and keyring is not None:
            try:
                val = keyring.get_password(self.app_name, key)
                if val:
                    with self._lock:
                        self._cache[key] = val
                    return val
            except Exception as e:
                logger.debug(f"keyring get failed: {e}")
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    data = json.load(f)
                if key in data:
                    raw = data[key]
                    if raw.startswith("encrypted:") and self._fernet:
                        raw = self._fernet.decrypt(raw[10:].encode()).decode()
                    with self._lock:
                        self._cache[key] = raw
                    return raw
            except Exception as e:
                logger.debug(f"config file read failed: {e}")
        return default

    def set(self, key: str, value: str, store_in: str = "keyring") -> None:
        with self._lock:
            self._cache[key] = value
        if store_in == "keyring" and self.use_keyring and keyring is not None:
            try:
                keyring.set_password(self.app_name, key, value)
                return
            except Exception as e:
                logger.warning(f"keyring set failed, falling back to file: {e}")
        config = {}
        if self.config_path.exists():
            with open(self.config_path) as f:
                config = json.load(f)
        config[key] = f"encrypted:{self._fernet.encrypt(value.encode()).decode()}" if self._fernet else value
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

    def delete(self, key: str) -> bool:
        deleted = False
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                deleted = True
        if self.use_keyring and keyring is not None:
            try:
                keyring.delete_password(self.app_name, key)
                deleted = True
            except Exception:
                pass
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    config = json.load(f)
                if key in config:
                    del config[key]
                    with open(self.config_path, "w") as f:
                        json.dump(config, f, indent=2)
                    deleted = True
            except Exception:
                pass
        env_key = f"{self.env_prefix}{key.upper()}"
        if env_key in os.environ:
            del os.environ[env_key]
            deleted = True
        return deleted

    def list_keys(self) -> Dict[str, str]:
        keys = {}
        for env_key in os.environ:
            if env_key.startswith(self.env_prefix):
                keys[env_key[len(self.env_prefix):].lower()] = "env"
        if self.config_path.exists():
            with open(self.config_path) as f:
                for k in json.load(f):
                    keys[k] = "config"
        return keys

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
        if self.config_path.exists():
            with open(self.config_path, "w") as f:
                json.dump({}, f)


credential_manager = CredentialManager()


def get_credential(key: str, default: Optional[str] = None) -> Optional[str]:
    return credential_manager.get(key, default)


def set_credential(key: str, value: str, store_in: str = "keyring") -> None:
    credential_manager.set(key, value, store_in)


def delete_credential(key: str) -> bool:
    return credential_manager.delete(key)
