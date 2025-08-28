# utils/jwt_auth.py
"""
Enhanced JWT authentication for the Universal Translation System.
This module provides tools for secure JWT authentication.
"""

import jwt
import time
import uuid
import logging
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .exceptions import AuthenticationError, AuthorizationError
from .credential_manager import get_credential

logger = logging.getLogger(__name__)

# Security scheme for FastAPI
security = HTTPBearer()

# Default settings
DEFAULT_ALGORITHM = "HS256"
DEFAULT_EXPIRATION = 30 * 60  # 30 minutes
DEFAULT_REFRESH_EXPIRATION = 7 * 24 * 60 * 60  # 7 days


class JWTAuth:
    """
    Enhanced JWT authentication manager.
    """
    
    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = DEFAULT_ALGORITHM,
        token_expiration: int = DEFAULT_EXPIRATION,
        refresh_expiration: int = DEFAULT_REFRESH_EXPIRATION,
        audience: Optional[str] = None,
        issuer: Optional[str] = None
    ):
        """
        Initialize JWT authentication manager.
        
        Args:
            secret_key: Secret key for signing tokens
            algorithm: JWT algorithm
            token_expiration: Token expiration time in seconds
            refresh_expiration: Refresh token expiration time in seconds
            audience: Expected audience for tokens
            issuer: Expected issuer for tokens
        """
        self.secret_key = secret_key or get_credential("jwt_secret")
        if not self.secret_key:
            raise AuthenticationError("JWT secret key not configured")
            
        self.algorithm = algorithm
        self.token_expiration = token_expiration
        self.refresh_expiration = refresh_expiration
        self.audience = audience
        self.issuer = issuer
        
        logger.debug(
            f"Initialized JWT auth with algorithm={algorithm}, "
            f"expiration={token_expiration}s"
        )
        
    def create_token(
        self,
        subject: str,
        payload: Optional[Dict[str, Any]] = None,
        expires_in: Optional[int] = None,
        audience: Optional[str] = None,
        issuer: Optional[str] = None,
        token_type: str = "access"
    ) -> str:
        """
        Create a JWT token.
        
        Args:
            subject: Subject (usually user ID)
            payload: Additional payload data
            expires_in: Expiration time in seconds
            audience: Token audience
            issuer: Token issuer
            token_type: Token type (access or refresh)
            
        Returns:
            JWT token
        """
        # Set expiration time
        if expires_in is None:
            expires_in = self.refresh_expiration if token_type == "refresh" else self.token_expiration
            
        # Create payload
        now = int(time.time())
        token_payload = {
            "sub": subject,
            "iat": now,
            "exp": now + expires_in,
            "jti": str(uuid.uuid4()),
            "type": token_type
        }
        
        # Add audience and issuer
        if audience or self.audience:
            token_payload["aud"] = audience or self.audience
        if issuer or self.issuer:
            token_payload["iss"] = issuer or self.issuer
            
        # Add custom payload
        if payload:
            token_payload.update(payload)
            
        # Create token
        return jwt.encode(
            token_payload,
            self.secret_key,
            algorithm=self.algorithm
        )
        
    def create_access_token(
        self,
        subject: str,
        payload: Optional[Dict[str, Any]] = None,
        expires_in: Optional[int] = None,
        audience: Optional[str] = None,
        issuer: Optional[str] = None
    ) -> str:
        """
        Create an access token.
        
        Args:
            subject: Subject (usually user ID)
            payload: Additional payload data
            expires_in: Expiration time in seconds
            audience: Token audience
            issuer: Token issuer
            
        Returns:
            Access token
        """
        return self.create_token(
            subject,
            payload,
            expires_in,
            audience,
            issuer,
            "access"
        )
        
    def create_refresh_token(
        self,
        subject: str,
        payload: Optional[Dict[str, Any]] = None,
        expires_in: Optional[int] = None,
        audience: Optional[str] = None,
        issuer: Optional[str] = None
    ) -> str:
        """
        Create a refresh token.
        
        Args:
            subject: Subject (usually user ID)
            payload: Additional payload data
            expires_in: Expiration time in seconds
            audience: Token audience
            issuer: Token issuer
            
        Returns:
            Refresh token
        """
        return self.create_token(
            subject,
            payload,
            expires_in,
            audience,
            issuer,
            "refresh"
        )
        
    def create_token_pair(
        self,
        subject: str,
        payload: Optional[Dict[str, Any]] = None,
        access_expires_in: Optional[int] = None,
        refresh_expires_in: Optional[int] = None,
        audience: Optional[str] = None,
        issuer: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Create an access token and refresh token pair.
        
        Args:
            subject: Subject (usually user ID)
            payload: Additional payload data
            access_expires_in: Access token expiration time in seconds
            refresh_expires_in: Refresh token expiration time in seconds
            audience: Token audience
            issuer: Token issuer
            
        Returns:
            Dictionary with access_token and refresh_token
        """
        access_token = self.create_access_token(
            subject,
            payload,
            access_expires_in,
            audience,
            issuer
        )
        
        refresh_token = self.create_refresh_token(
            subject,
            payload,
            refresh_expires_in,
            audience,
            issuer
        )
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }
        
    def decode_token(
        self,
        token: str,
        verify_exp: bool = True,
        verify_aud: bool = True,
        verify_iss: bool = True,
        expected_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Decode and verify a JWT token.
        
        Args:
            token: JWT token
            verify_exp: Whether to verify expiration
            verify_aud: Whether to verify audience
            verify_iss: Whether to verify issuer
            expected_type: Expected token type
            
        Returns:
            Token payload
            
        Raises:
            AuthenticationError: If token verification fails
        """
        try:
            # Set verification options
            options = {
                "verify_exp": verify_exp,
                "verify_aud": verify_aud and bool(self.audience),
                "verify_iss": verify_iss and bool(self.issuer)
            }
            
            # Set audience and issuer
            kwargs = {}
            if verify_aud and self.audience:
                kwargs["audience"] = self.audience
            if verify_iss and self.issuer:
                kwargs["issuer"] = self.issuer
                
            # Decode token
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options=options,
                **kwargs
            )
            
            # Verify token type
            if expected_type and payload.get("type") != expected_type:
                raise AuthenticationError(f"Invalid token type: {payload.get('type')}")
                
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidAudienceError:
            raise AuthenticationError("Token has invalid audience")
        except jwt.InvalidIssuerError:
            raise AuthenticationError("Token has invalid issuer")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {str(e)}")
            
    def refresh_access_token(
        self,
        refresh_token: str,
        payload: Optional[Dict[str, Any]] = None,
        expires_in: Optional[int] = None,
        audience: Optional[str] = None,
        issuer: Optional[str] = None
    ) -> str:
        """
        Refresh an access token using a refresh token.
        
        Args:
            refresh_token: Refresh token
            payload: Additional payload data
            expires_in: Expiration time in seconds
            audience: Token audience
            issuer: Token issuer
            
        Returns:
            New access token
            
        Raises:
            AuthenticationError: If refresh token is invalid
        """
        # Decode refresh token
        refresh_payload = self.decode_token(
            refresh_token,
            expected_type="refresh"
        )
        
        # Create new access token
        subject = refresh_payload["sub"]
        
        # Merge payloads
        merged_payload = {}
        if refresh_payload:
            # Copy payload from refresh token, excluding standard claims
            for key, value in refresh_payload.items():
                if key not in ("sub", "exp", "iat", "jti", "type", "aud", "iss"):
                    merged_payload[key] = value
                    
        # Add custom payload
        if payload:
            merged_payload.update(payload)
            
        return self.create_access_token(
            subject,
            merged_payload,
            expires_in,
            audience,
            issuer
        )
        
    def get_token_dependency(
        self,
        verify_exp: bool = True,
        verify_aud: bool = True,
        verify_iss: bool = True,
        expected_type: Optional[str] = None
    ) -> Callable:
        """
        Get a FastAPI dependency for JWT authentication.
        
        Args:
            verify_exp: Whether to verify expiration
            verify_aud: Whether to verify audience
            verify_iss: Whether to verify issuer
            expected_type: Expected token type
            
        Returns:
            FastAPI dependency function
        """
        async def dependency(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
            try:
                return self.decode_token(
                    credentials.credentials,
                    verify_exp,
                    verify_aud,
                    verify_iss,
                    expected_type
                )
            except AuthenticationError as e:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=str(e),
                    headers={"WWW-Authenticate": "Bearer"}
                )
                
        return dependency
        
    def get_user_dependency(
        self,
        verify_exp: bool = True,
        verify_aud: bool = True,
        verify_iss: bool = True,
        expected_type: Optional[str] = "access"
    ) -> Callable:
        """
        Get a FastAPI dependency for JWT authentication that returns the user ID.
        
        Args:
            verify_exp: Whether to verify expiration
            verify_aud: Whether to verify audience
            verify_iss: Whether to verify issuer
            expected_type: Expected token type
            
        Returns:
            FastAPI dependency function
        """
        token_dependency = self.get_token_dependency(
            verify_exp,
            verify_aud,
            verify_iss,
            expected_type
        )
        
        async def dependency(payload: Dict[str, Any] = Depends(token_dependency)) -> str:
            return payload["sub"]
            
        return dependency
        
    def get_scopes_dependency(
        self,
        required_scopes: List[str],
        verify_exp: bool = True,
        verify_aud: bool = True,
        verify_iss: bool = True,
        expected_type: Optional[str] = "access"
    ) -> Callable:
        """
        Get a FastAPI dependency for JWT authentication that checks scopes.
        
        Args:
            required_scopes: List of required scopes
            verify_exp: Whether to verify expiration
            verify_aud: Whether to verify audience
            verify_iss: Whether to verify issuer
            expected_type: Expected token type
            
        Returns:
            FastAPI dependency function
        """
        token_dependency = self.get_token_dependency(
            verify_exp,
            verify_aud,
            verify_iss,
            expected_type
        )
        
        async def dependency(payload: Dict[str, Any] = Depends(token_dependency)) -> Dict[str, Any]:
            # Get token scopes
            token_scopes = payload.get("scopes", [])
            
            # Check if token has all required scopes
            for scope in required_scopes:
                if scope not in token_scopes:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Missing required scope: {scope}",
                        headers={"WWW-Authenticate": "Bearer"}
                    )
                    
            return payload
            
        return dependency

    def blacklist_token(self, token: str, reason: str = "") -> None:
        """
        Blacklist a token.
    
        Args:
            token: JWT token
            reason: Reason for blacklisting
        """
        try:
            # Decode token without verification
            payload = jwt.decode(
                token,
                options={"verify_signature": False}
            )
        
            # Get token ID
            jti = payload.get("jti")
            if not jti:
                logger.warning("Token has no JTI, cannot blacklist")
                return
            
            # Add to blacklist
            self._blacklist[jti] = {
                "timestamp": time.time(),
                "reason": reason,
                "exp": payload.get("exp")
            }
         
            # Clean expired blacklist entries
            self._clean_blacklist()
        except Exception as e:
            logger.error(f"Failed to blacklist token: {e}")
        
    def is_blacklisted(self, token: str) -> bool:
        """
        Check if a token is blacklisted.
    
        Args:
            token: JWT token
        
        Returns:
            True if the token is blacklisted
        """
        try:
           # Decode token without verification
            payload = jwt.decode(
               token,
               options={"verify_signature": False}
            )
        
            # Get token ID
            jti = payload.get("jti")
            if not jti:
                return False
            
            # Check blacklist
            return jti in self._blacklist
        except Exception as e:
            logger.error(f"Failed to check blacklist: {e}")
            return False
        
    def _clean_blacklist(self) -> None:
        """Clean expired blacklist entries."""
        now = time.time()
        expired = [
            jti for jti, data in self._blacklist.items()
            if data.get("exp") and data["exp"] < now
        ]
    
        for jti in expired:
            del self._blacklist[jti]

    
    def introspect_token(self, token: str) -> Dict[str, Any]:
        """
        Introspect a token.
    
        Args:
           token: JWT token
        
        Returns:
            Token information
        """
        try:
            # Decode token
            payload = self.decode_token(token, verify_exp=False)
        
            # Check if token is active
            now = int(time.time())
            is_active = (
               not self.is_blacklisted(token) and
               payload.get("exp", 0) > now
            )
        
            return {
                "active": is_active,
                "sub": payload.get("sub"),
                "exp": payload.get("exp"),
                "iat": payload.get("iat"),
                "jti": payload.get("jti"),
                "type": payload.get("type"),
                "scopes": payload.get("scopes", []),
                "is_expired": payload.get("exp", 0) <= now,
                "is_blacklisted": self.is_blacklisted(token)
            }
        except Exception as e:
            return {
                "active": False,
                "error": str(e)
            }

    def create_encrypted_token(
        self,
        subject: str,
        payload: Optional[Dict[str, Any]] = None,
        expires_in: Optional[int] = None,
        audience: Optional[str] = None,
        issuer: Optional[str] = None,
        token_type: str = "access",
        encryption_key: Optional[str] = None
    ) -> str:
        """
        Create an encrypted JWT token.
    
        Args:
            subject: Subject (usually user ID)
            payload: Additional payload data
            expires_in: Expiration time in seconds
            audience: Token audience
            issuer: Token issuer
            token_type: Token type (access or refresh)
            encryption_key: Key for encryption
        
        Returns:
            Encrypted JWT token
        """
        # Create token
        token = self.create_token(
            subject,
            payload,
            expires_in,
            audience,
            issuer,
            token_type
        )
    
        # Encrypt token
        if not encryption_key:
            encryption_key = self.secret_key
        
        from cryptography.fernet import Fernet
        import base64
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    
        # Derive key
        salt = b'UniversalTranslationSystem'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000
        )
        key = base64.urlsafe_b64encode(kdf.derive(encryption_key.encode()))
    
        # Encrypt
        f = Fernet(key)
        encrypted = f.encrypt(token.encode()).decode()
    
        return f"ENC:{encrypted}"
    
    def decode_encrypted_token(
        self,
        token: str,
        verify_exp: bool = True,
        verify_aud: bool = True,
        verify_iss: bool = True,
        expected_type: Optional[str] = None,
        encryption_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Decode and verify an encrypted JWT token.
    
        Args:
            token: Encrypted JWT token
            verify_exp: Whether to verify expiration
            verify_aud: Whether to verify audience
            verify_iss: Whether to verify issuer
            expected_type: Expected token type
            encryption_key: Key for decryption
        
        Returns:
            Token payload
        
        Raises:
            AuthenticationError: If token verification fails
        """
        # Check if token is encrypted
        if not token.startswith("ENC:"):
            raise AuthenticationError("Token is not encrypted")
        
        # Decrypt token
        encrypted = token[4:]
    
        if not encryption_key:
            encryption_key = self.secret_key
        
        from cryptography.fernet import Fernet
        import base64
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    
        # Derive key
        salt = b'UniversalTranslationSystem'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000
        )
        key = base64.urlsafe_b64encode(kdf.derive(encryption_key.encode()))
    
        # Decrypt
        try:
            f = Fernet(key)
            decrypted = f.decrypt(encrypted.encode()).decode()
        except Exception as e:
            raise AuthenticationError(f"Failed to decrypt token: {e}")
        
        # Decode token
        return self.decode_token(
            decrypted,
            verify_exp,
            verify_aud,
            verify_iss,
            expected_type
        )

# Create a global JWT auth instance
jwt_auth = JWTAuth()


def require_auth(
    verify_exp: bool = True,
    verify_aud: bool = True,
    verify_iss: bool = True,
    expected_type: Optional[str] = "access"
) -> Callable:
    """
    Decorator for requiring JWT authentication.
    
    Args:
        verify_exp: Whether to verify expiration
        verify_aud: Whether to verify audience
        verify_iss: Whether to verify issuer
        expected_type: Expected token type
        
    Returns:
        Decorator function
    """
    return jwt_auth.get_token_dependency(
        verify_exp,
        verify_aud,
        verify_iss,
        expected_type
    )


def require_user(
    verify_exp: bool = True,
    verify_aud: bool = True,
    verify_iss: bool = True,
    expected_type: Optional[str] = "access"
) -> Callable:
    """
    Decorator for requiring JWT authentication and returning the user ID.
    
    Args:
        verify_exp: Whether to verify expiration
        verify_aud: Whether to verify audience
        verify_iss: Whether to verify issuer
        expected_type: Expected token type
        
    Returns:
        Decorator function
    """
    return jwt_auth.get_user_dependency(
        verify_exp,
        verify_aud,
        verify_iss,
        expected_type
    )


def require_scopes(
    required_scopes: List[str],
    verify_exp: bool = True,
    verify_aud: bool = True,
    verify_iss: bool = True,
    expected_type: Optional[str] = "access"
) -> Callable:
    """
    Decorator for requiring JWT authentication with specific scopes.
    
    Args:
        required_scopes: List of required scopes
        verify_exp: Whether to verify expiration
        verify_aud: Whether to verify audience
        verify_iss: Whether to verify issuer
        expected_type: Expected token type
        
    Returns:
        Decorator function
    """
    return jwt_auth.get_scopes_dependency(
        required_scopes,
        verify_exp,
        verify_aud,
        verify_iss,
        expected_type
    )
