# universal-decoder-node/universal_decoder_node/utils/auth.py
import os
import logging
import time
import jwt
from typing import Dict, Optional, List, Tuple

logger = logging.getLogger(__name__)

class APIKeyManager:
    """
    Manages API keys for authentication.
    
    Features:
    - JWT token validation
    - API key rotation
    - Rate limiting integration
    """
    
    def __init__(self, secret_key: Optional[str] = None):
        """
        Initialize the API key manager.
        
        Args:
            secret_key: Secret key for JWT token signing. If None, uses environment variable.
        """
        self.secret_key = secret_key or os.environ.get("DECODER_JWT_SECRET", None)
        if not self.secret_key:
            # Generate a secure random key if none is provided
            import secrets
            self.secret_key = secrets.token_hex(32)
            logger.warning("No JWT secret key provided, generated a random one. This will not persist across restarts.")
        
        # Store active API keys with metadata
        self.api_keys: Dict[str, Dict] = {}
        
        # Load API keys from environment if available
        self._load_api_keys_from_env()
    
    def _load_api_keys_from_env(self):
        """Load API keys from environment variables"""
        # Format: DECODER_API_KEY_NAME=key:role:expiry
        for key, value in os.environ.items():
            if key.startswith("DECODER_API_KEY_"):
                try:
                    name = key[len("DECODER_API_KEY_"):].lower()
                    parts = value.split(":")
                    if len(parts) >= 1:
                        api_key = parts[0]
                        role = parts[1] if len(parts) > 1 else "user"
                        expiry = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None
                        
                        self.api_keys[api_key] = {
                            "name": name,
                            "role": role,
                            "expiry": expiry
                        }
                        logger.info(f"Loaded API key '{name}' with role '{role}'")
                except Exception as e:
                    logger.error(f"Failed to parse API key from environment: {e}")
    
    def validate_key(self, api_key: str) -> Tuple[bool, Optional[Dict]]:
        """
        Validate an API key.
        
        Args:
            api_key: API key to validate
            
        Returns:
            Tuple of (is_valid, metadata)
        """
        if not api_key:
            return False, None
            
        # Check if it's a direct API key
        if api_key in self.api_keys:
            metadata = self.api_keys[api_key]
            
            # Check expiry
            if metadata.get("expiry") and metadata["expiry"] < time.time():
                logger.warning(f"API key '{metadata['name']}' has expired")
                return False, None
                
            return True, metadata
            
        # Check if it's a JWT token
        try:
            payload = jwt.decode(api_key, self.secret_key, algorithms=["HS256"])
            return True, payload
        except jwt.PyJWTError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return False, None
    
    def create_token(self, subject: str, role: str = "user", expiry: Optional[int] = None) -> str:
        """
        Create a JWT token.
        
        Args:
            subject: Subject of the token (usually username or client ID)
            role: Role of the token (e.g., "admin", "user")
            expiry: Expiration time in seconds from now, or None for no expiry
            
        Returns:
            JWT token
        """
        payload = {
            "sub": subject,
            "role": role,
            "iat": int(time.time())
        }
        
        if expiry:
            payload["exp"] = int(time.time()) + expiry
            
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    def add_api_key(self, name: str, role: str = "user", expiry: Optional[int] = None) -> str:
        """
        Add a new API key.
        
        Args:
            name: Name of the API key
            role: Role of the API key
            expiry: Expiration time in seconds from now, or None for no expiry
            
        Returns:
            The generated API key
        """
        import secrets
        api_key = secrets.token_hex(16)
        
        expiry_time = int(time.time()) + expiry if expiry else None
        
        self.api_keys[api_key] = {
            "name": name,
            "role": role,
            "expiry": expiry_time
        }
        
        return api_key
    
    def revoke_api_key(self, api_key: str) -> bool:
        """
        Revoke an API key.
        
        Args:
            api_key: API key to revoke
            
        Returns:
            True if the key was revoked, False otherwise
        """
        if api_key in self.api_keys:
            del self.api_keys[api_key]
            return True
        return False