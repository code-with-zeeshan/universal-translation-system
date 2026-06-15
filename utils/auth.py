# utils/auth.py
import secrets
import hashlib
import hmac
import json
import threading
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging
from datetime import datetime
import os

from .credential_manager import credential_manager

logger = logging.getLogger(__name__)


class APIKeyManager:
    """Manage API keys for authentication with optional encrypted storage."""

    def __init__(self, keys_file: str = "config/api_keys.json"):
        self.keys_file = Path(keys_file)
        # Toggle to store API key metadata via CredentialManager (encrypted config)
        self.use_credmgr = os.environ.get("UTS_API_KEYS_USE_CREDMGR", "false").lower() == "true"
        self.keys = self._load_keys()
        self._lock = threading.RLock()

    def _load_keys(self) -> Dict[str, Dict[str, any]]:
        """Load API keys from secure storage or file."""
        if self.use_credmgr:
            try:
                data = credential_manager.get("api_keys_json")
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.warning(f"Failed to load API keys from credential manager: {e}")
            # Fallback to file if present
        if self.keys_file.exists():
            try:
                with open(self.keys_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read API keys file: {e}")
        return {}

    def _save_keys(self):
        """Save API keys to secure storage or file."""
        if self.use_credmgr:
            try:
                credential_manager.set("api_keys_json", json.dumps(self.keys), store_in="config")
                return
            except Exception as e:
                logger.error(f"Failed to save API keys via credential manager: {e}")
                # Do not fallback to plaintext file when encrypted path is requested
                raise
        # File-based storage (plaintext metadata only; keys themselves are never stored)
        self.keys_file.parent.mkdir(exist_ok=True)
        with open(self.keys_file, 'w', encoding='utf-8') as f:
            json.dump(self.keys, f, indent=2)

    def generate_api_key(self, client_name: str, permissions: List[str]) -> str:
        """Generate and store a new API key (returns the plaintext once)."""
        api_key = secrets.token_urlsafe(32)
        lookup = hashlib.sha256(api_key.encode()).hexdigest()
        salt = secrets.token_bytes(16)
        key_hash = hashlib.pbkdf2_hmac('sha256', api_key.encode(), salt=salt, iterations=600000).hex()
        now = datetime.now().isoformat()
        with self._lock:
            self.keys[lookup] = {
                'salt': salt.hex(),
                'hash': key_hash,
                'client_name': client_name,
                'permissions': permissions,
                'created_at': now,
                'last_used': None,
                'request_count': 0,
                'rotated_at': None,
            }
            self._save_keys()
        logger.info(f"Generated API key for {client_name}")
        return api_key

    def validate_api_key(self, api_key: str) -> Optional[Dict[str, any]]:
        """Validate API key and return metadata."""
        lookup = hashlib.sha256(api_key.encode()).hexdigest()
        with self._lock:
            if lookup in self.keys:
                meta = self.keys[lookup]
                salt = bytes.fromhex(meta.get("salt", ""))
                key_hash = hashlib.pbkdf2_hmac('sha256', api_key.encode(), salt=salt, iterations=600000).hex()
                if not hmac.compare_digest(key_hash, meta["hash"]):
                    return None
                meta['last_used'] = datetime.now().isoformat()
                meta['request_count'] += 1
                self._save_keys()
                return dict(meta)
        return None

    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key; returns True if revoked."""
        lookup = hashlib.sha256(api_key.encode()).hexdigest()
        with self._lock:
            if lookup in self.keys:
                client = self.keys[lookup]['client_name']
                del self.keys[lookup]
                self._save_keys()
                logger.info(f"Revoked API key for {client}")
                return True
        return False

    def rotate_api_key(self, api_key: str) -> Tuple[bool, Optional[str]]:
        """
        Rotate an existing API key: revoke old, issue new for same client/permissions.
        Returns (success, new_api_key_or_none).
        """
        lookup = hashlib.sha256(api_key.encode()).hexdigest()
        with self._lock:
            meta = self.keys.get(lookup)
            if not meta:
                return False, None
            client = meta['client_name']
            perms = meta['permissions']
            meta['rotated_at'] = datetime.now().isoformat()
            self._save_keys()
            del self.keys[lookup]
            new_key = self.generate_api_key(client, perms)
        return True, new_key