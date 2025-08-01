# utils/auth.py
import secrets
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

class APIKeyManager:
    """Manage API keys for authentication"""
    
    def __init__(self, keys_file: str = "config/api_keys.json"):
        self.keys_file = Path(keys_file)
        self.keys = self._load_keys()
    
    def _load_keys(self) -> Dict[str, Dict[str, any]]:
        """Load API keys from file"""
        if self.keys_file.exists():
            with open(self.keys_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_keys(self):
        """Save API keys to file"""
        self.keys_file.parent.mkdir(exist_ok=True)
        with open(self.keys_file, 'w') as f:
            json.dump(self.keys, f, indent=2)
    
    def generate_api_key(self, client_name: str, permissions: List[str]) -> str:
        """Generate new API key"""
        # Generate secure random key
        api_key = secrets.token_urlsafe(32)
        
        # Hash for storage
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Store key info
        self.keys[key_hash] = {
            'client_name': client_name,
            'permissions': permissions,
            'created_at': datetime.now().isoformat(),
            'last_used': None,
            'request_count': 0
        }
        
        self._save_keys()
        logger.info(f"Generated API key for {client_name}")
        
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, any]]:
        """Validate API key and return permissions"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        if key_hash in self.keys:
            # Update usage stats
            self.keys[key_hash]['last_used'] = datetime.now().isoformat()
            self.keys[key_hash]['request_count'] += 1
            self._save_keys()
            
            return self.keys[key_hash]
        
        return None
    
    def revoke_api_key(self, api_key: str):
        """Revoke an API key"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        if key_hash in self.keys:
            client = self.keys[key_hash]['client_name']
            del self.keys[key_hash]
            self._save_keys()
            logger.info(f"Revoked API key for {client}")