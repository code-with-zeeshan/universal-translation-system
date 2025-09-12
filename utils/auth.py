# utils/auth.py
import secrets
import hashlib
import json
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
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        now = datetime.now().isoformat()
        self.keys[key_hash] = {
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
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        if key_hash in self.keys:
            self.keys[key_hash]['last_used'] = datetime.now().isoformat()
            self.keys[key_hash]['request_count'] += 1
            self._save_keys()
            return self.keys[key_hash]
        return None

    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key; returns True if revoked."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        if key_hash in self.keys:
            client = self.keys[key_hash]['client_name']
            del self.keys[key_hash]
            self._save_keys()
            logger.info(f"Revoked API key for {client}")
            return True
        return False

    def rotate_api_key(self, api_key: str) -> Tuple[bool, Optional[str]]:
        """
        Rotate an existing API key: revoke old, issue new for same client/permissions.
        Returns (success, new_api_key_or_none).
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        meta = self.keys.get(key_hash)
        if not meta:
            return False, None
        client = meta['client_name']
        perms = meta['permissions']
        # Mark rotated time for old record then remove
        meta['rotated_at'] = datetime.now().isoformat()
        # Persist rotation mark then delete
        self._save_keys()
        del self.keys[key_hash]
        # Issue new key
        new_key = self.generate_api_key(client, perms)
        return True, new_key