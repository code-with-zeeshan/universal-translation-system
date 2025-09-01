# utils/credential_manager.py
"""
Secure credential management for the Universal Translation System.
This module provides tools for securely managing credentials.
"""

import os
import json
import base64
import logging
import threading
from typing import Dict, Any, Optional, Union, Callable, Tuple, List
import time
from pathlib import Path
# Optional dependencies: keyring and cryptography
try:
    import keyring  # type: ignore
except Exception:  # pragma: no cover
    keyring = None  # type: ignore
try:
    from cryptography.fernet import Fernet  # type: ignore
    from cryptography.hazmat.primitives import hashes  # type: ignore
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC  # type: ignore
except Exception:  # pragma: no cover
    Fernet = None  # type: ignore
    hashes = None  # type: ignore
    PBKDF2HMAC = None  # type: ignore
from .exceptions import SecurityError

logger = logging.getLogger(__name__)


class CredentialManager:
    """
    Secure credential manager that supports multiple storage backends.
    """
    
    def __init__(
        self,
        app_name: str = "UniversalTranslationSystem",
        env_prefix: str = "UTS_",
        config_path: Optional[str] = None,
        use_keyring: bool = True,
        encryption_key: Optional[str] = None
    ):
        """
        Initialize credential manager.
        
        Args:
            app_name: Application name for keyring
            env_prefix: Prefix for environment variables
            config_path: Path to configuration file
            use_keyring: Whether to use the system keyring
            encryption_key: Key for encrypting credentials in the config file
        """
        self.app_name = app_name
        self.env_prefix = env_prefix
        self.use_keyring = use_keyring
        
        # Set config path
        if config_path:
            self.config_path = Path(config_path)
        else:
            home_dir = Path.home()
            self.config_path = home_dir / f".{app_name.lower()}" / "credentials.json"
            
        # Initialize encryption
        self.encryption_key = encryption_key
        if encryption_key and Fernet and PBKDF2HMAC and hashes:
            self._init_encryption(encryption_key)
        else:
            if encryption_key and not (Fernet and PBKDF2HMAC and hashes):
                logger.warning("Encryption requested but cryptography is not installed; proceeding without encryption")
            self.fernet = None
            
        # Cache for credentials
        self._cache: Dict[str, str] = {}
        self._cache_lock = threading.RLock()
        
        # Create config directory if it doesn't exist
        if not self.config_path.parent.exists():
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
        logger.debug(f"Initialized credential manager with config at {self.config_path}")
        
    def _init_encryption(self, key: str) -> None:
        """Initialize encryption with the given key."""
        # Derive a key from the password
        salt = b'UniversalTranslationSystem'  # Fixed salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000
        )
        derived_key = base64.urlsafe_b64encode(kdf.derive(key.encode()))
        self.fernet = Fernet(derived_key)
        
    def _encrypt(self, value: str) -> str:
        """Encrypt a value."""
        if not self.fernet:
            return value
        return self.fernet.encrypt(value.encode()).decode()
        
    def _decrypt(self, value: str) -> str:
        """Decrypt a value."""
        if not self.fernet:
            return value
        try:
            return self.fernet.decrypt(value.encode()).decode()
        except Exception as e:
            logger.error(f"Failed to decrypt value: {e}")
            raise SecurityError("Failed to decrypt credential")
            
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a credential.
        
        Args:
            key: Credential key
            default: Default value if not found
            
        Returns:
            Credential value or default
        """
        # Check cache first
        with self._cache_lock:
            if key in self._cache:
                return self._cache[key]
                
        # Try environment variable
        env_key = f"{self.env_prefix}{key.upper()}"
        env_value = os.environ.get(env_key)
        if env_value:
            with self._cache_lock:
                self._cache[key] = env_value
            return env_value
            
        # Try keyring
        if self.use_keyring and keyring is not None:
            try:
                keyring_value = keyring.get_password(self.app_name, key)
                if keyring_value:
                    with self._cache_lock:
                        self._cache[key] = keyring_value
                    return keyring_value
            except Exception as e:
                logger.warning(f"Failed to get credential from keyring: {e}")
                
        # Try config file
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    
                if key in config:
                    value = config[key]
                    if value.startswith('encrypted:'):
                        value = self._decrypt(value[10:])
                        
                    with self._cache_lock:
                        self._cache[key] = value
                    return value
            except Exception as e:
                logger.warning(f"Failed to get credential from config file: {e}")
                
        return default
        
    def set(self, key: str, value: str, store_in: str = "keyring") -> None:
        """
        Set a credential.
        
        Args:
            key: Credential key
            value: Credential value
            store_in: Where to store the credential (keyring, config, env)
        """
        # Update cache
        with self._cache_lock:
            self._cache[key] = value
            
        # Store in keyring
        if store_in == "keyring" and self.use_keyring and keyring is not None:
            try:
                keyring.set_password(self.app_name, key, value)
                logger.debug(f"Stored credential {key} in keyring")
                return
            except Exception as e:
                logger.warning(f"Failed to store credential in keyring: {e}")
                
        # Store in config file
        if store_in == "config":
            config = {}
            if self.config_path.exists():
                try:
                    with open(self.config_path, 'r') as f:
                        config = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to read config file: {e}")
                    
            # Encrypt value if encryption is enabled
            if self.fernet:
                config[key] = f"encrypted:{self._encrypt(value)}"
            else:
                config[key] = value
                
            try:
                with open(self.config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                logger.debug(f"Stored credential {key} in config file")
                return
            except Exception as e:
                logger.warning(f"Failed to write config file: {e}")
                
        # Store in environment variable
        if store_in == "env":
            env_key = f"{self.env_prefix}{key.upper()}"
            os.environ[env_key] = value
            logger.debug(f"Stored credential {key} in environment variable")
            return
            
        raise ValueError(f"Invalid storage location: {store_in}")
        
    def delete(self, key: str) -> bool:
        """
        Delete a credential.
        
        Args:
            key: Credential key
            
        Returns:
            True if the credential was deleted
        """
        deleted = False
        
        # Remove from cache
        with self._cache_lock:
            if key in self._cache:
                del self._cache[key]
                deleted = True
                
        # Remove from keyring
        if self.use_keyring and keyring is not None:
            try:
                keyring.delete_password(self.app_name, key)
                deleted = True
            except Exception as e:
                logger.debug(f"Failed to delete credential from keyring: {e}")
                
        # Remove from config file
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    
                if key in config:
                    del config[key]
                    with open(self.config_path, 'w') as f:
                        json.dump(config, f, indent=2)
                    deleted = True
            except Exception as e:
                logger.warning(f"Failed to delete credential from config file: {e}")
                
        # Remove from environment variable
        env_key = f"{self.env_prefix}{key.upper()}"
        if env_key in os.environ:
            del os.environ[env_key]
            deleted = True
            
        return deleted
        
    def list_keys(self) -> Dict[str, str]:
        """
        List all credential keys and their storage locations.
        
        Returns:
            Dictionary mapping keys to storage locations
        """
        keys = {}
        
        # Check environment variables
        for env_key in os.environ:
            if env_key.startswith(self.env_prefix):
                key = env_key[len(self.env_prefix):].lower()
                keys[key] = "env"
                
        # Check config file
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    
                for key in config:
                    keys[key] = "config"
            except Exception as e:
                logger.warning(f"Failed to read config file: {e}")
                
        # Check keyring
        if self.use_keyring:
            # Keyring doesn't provide a way to list all keys
            # We can only check if specific keys exist
            pass
            
        return keys
        
    def clear(self) -> None:
        """Clear all credentials."""
        # Clear cache
        with self._cache_lock:
            self._cache.clear()
            
        # Clear config file
        if self.config_path.exists():
            try:
                with open(self.config_path, 'w') as f:
                    json.dump({}, f)
            except Exception as e:
                logger.warning(f"Failed to clear config file: {e}")
                
        # We can't clear keyring or environment variables globally
        # as they may be used by other applications
        
    def get_dict(self, prefix: str = "") -> Dict[str, str]:
        """
        Get all credentials with the given prefix as a dictionary.
        
        Args:
            prefix: Credential key prefix
            
        Returns:
            Dictionary of credentials
        """
        result = {}
        
        # Get all keys
        keys = self.list_keys()
        
        # Get values for keys with the prefix
        for key in keys:
            if key.startswith(prefix):
                value = self.get(key)
                if value is not None:
                    result[key] = value
                    
        return result

    def rotate_credential(self, key: str, generator: Callable[[], str], 
                         store_in: str = "keyring") -> Tuple[str, str]:
        """
        Rotate a credential.
    
        Args:
            key: Credential key
            generator: Function that generates a new credential
            store_in: Where to store the new credential
        
        Returns:
            Tuple of (old_value, new_value)
        """
        # Get old value
        old_value = self.get(key)
    
        # Generate new value
        new_value = generator()
    
        # Store new value
        self.set(key, new_value, store_in)
    
        return old_value, new_value

    def set_with_expiration(self, key: str, value: str, 
                            expires_in: int, store_in: str = "keyring") -> None:
        """
        Set a credential with expiration.
    
        Args:
            key: Credential key
            value: Credential value
            expires_in: Expiration time in seconds
            store_in: Where to store the credential
        """
        # Store expiration time
        expiration_key = f"{key}__expires_at"
        expires_at = int(time.time()) + expires_in
    
        # Store credential and expiration
        self.set(key, value, store_in)
        self.set(expiration_key, str(expires_at), store_in)
    
    def get_with_expiration(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a credential, checking expiration.
    
        Args:
            key: Credential key
            default: Default value if not found or expired
        
        Returns:
            Credential value or default
        """
        # Check expiration
        expiration_key = f"{key}__expires_at"
        expires_at_str = self.get(expiration_key)
    
        if expires_at_str:
            try:
                expires_at = int(expires_at_str)
                if time.time() > expires_at:
                    # Credential has expired
                    self.delete(key)
                    self.delete(expiration_key)
                    return default
            except ValueError:
                pass
            
        # Get credential
        return self.get(key, default)

    def export_credentials(self, include_keys: Optional[List[str]] = None,
                          exclude_keys: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Export credentials.
    
        Args:
            include_keys: Keys to include (None for all)
            exclude_keys: Keys to exclude
        
        Returns:
            Dictionary of credentials
        """
        # Get all keys
        keys = self.list_keys()
    
        # Filter keys
        if include_keys:
            keys = {k: v for k, v in keys.items() if k in include_keys}
        if exclude_keys:
            keys = {k: v for k, v in keys.items() if k not in exclude_keys}
        
        # Get values
        result = {}
        for key in keys:
            value = self.get(key)
            if value is not None:
                result[key] = value
            
        return result
    
    def import_credentials(self, credentials: Dict[str, str], 
                          store_in: str = "keyring") -> None:
        """
        Import credentials.
    
        Args:
            credentials: Dictionary of credentials
            store_in: Where to store the credentials
        """
        for key, value in credentials.items():
            self.set(key, value, store_in)
    
    
    
# Create a global credential manager instance
credential_manager = CredentialManager()


def get_credential(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get a credential.
    
    Args:
        key: Credential key
        default: Default value if not found
        
    Returns:
        Credential value or default
    """
    return credential_manager.get(key, default)


def set_credential(key: str, value: str, store_in: str = "keyring") -> None:
    """
    Set a credential.
    
    Args:
        key: Credential key
        value: Credential value
        store_in: Where to store the credential (keyring, config, env)
    """
    credential_manager.set(key, value, store_in)


def delete_credential(key: str) -> bool:
    """
    Delete a credential.
    
    Args:
        key: Credential key
        
    Returns:
        True if the credential was deleted
    """
    return credential_manager.delete(key)
