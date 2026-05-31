# utils/jwt_auth.py
"""
JWT authentication using PyJWT and FastAPI security.
Minimal wrapper around PyJWT + FastAPI HTTPBearer.
"""

import os
import time
import uuid
import logging
from typing import Dict, Any, Optional, List, Callable

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

security = HTTPBearer()

# Resolve algorithm from env
_alg = os.getenv("UTS_JWT_DEFAULT_ALG", "RS256" if (os.getenv("JWT_PUBLIC_KEY") or os.getenv("JWT_PUBLIC_KEY_PATH")) else "HS256")
DEFAULT_ALGORITHM = _alg
DEFAULT_EXPIRATION = 30 * 60
DEFAULT_REFRESH_EXPIRATION = 7 * 24 * 60 * 60


class JWTAuth:
    def __init__(self, secret_key: Optional[str] = None, algorithm: str = DEFAULT_ALGORITHM,
                 token_expiration: int = DEFAULT_EXPIRATION, refresh_expiration: int = DEFAULT_REFRESH_EXPIRATION,
                 audience: Optional[str] = None, issuer: Optional[str] = None):
        self.algorithm = algorithm
        self.token_expiration = token_expiration
        self.refresh_expiration = refresh_expiration
        self.audience = audience
        self.issuer = issuer
        self.secret_key = secret_key or os.getenv("JWT_SECRET") or os.getenv("COORDINATOR_JWT_SECRET") or os.getenv("DECODER_JWT_SECRET")
        self.public_keys: List[str] = []
        if algorithm.startswith("RS"):
            for src in [os.getenv("JWT_PUBLIC_KEY", ""), *(open(p).read() for p in os.getenv("JWT_PUBLIC_KEY_PATH", "").split("||") if p)]:
                if src.strip():
                    self.public_keys.append(src.strip())
        if not self.secret_key and not self.public_keys:
            raise ValueError("JWT keys not configured")

    def _payload(self, subject: str, expires_in: int, token_type: str, payload: Optional[Dict] = None) -> Dict:
        now = int(time.time())
        data = {"sub": subject, "iat": now, "exp": now + expires_in, "jti": str(uuid.uuid4()), "type": token_type}
        if self.audience:
            data["aud"] = self.audience
        if self.issuer:
            data["iss"] = self.issuer
        if payload:
            data.update(payload)
        return data

    def create_token(self, subject: str, payload: Optional[Dict] = None,
                     expires_in: Optional[int] = None, audience: Optional[str] = None,
                     issuer: Optional[str] = None, token_type: str = "access") -> str:
        exp = expires_in or (self.refresh_expiration if token_type == "refresh" else self.token_expiration)
        data = self._payload(subject, exp, token_type, payload)
        if audience:
            data["aud"] = audience
        if issuer:
            data["iss"] = issuer
        return jwt.encode(data, self.secret_key, algorithm=self.algorithm)

    def create_access_token(self, subject: str, payload: Optional[Dict] = None,
                            expires_in: Optional[int] = None, audience: Optional[str] = None,
                            issuer: Optional[str] = None) -> str:
        return self.create_token(subject, payload, expires_in, audience, issuer, "access")

    def create_refresh_token(self, subject: str, payload: Optional[Dict] = None,
                             expires_in: Optional[int] = None, audience: Optional[str] = None,
                             issuer: Optional[str] = None) -> str:
        return self.create_token(subject, payload, expires_in, audience, issuer, "refresh")

    def create_token_pair(self, subject: str, payload: Optional[Dict] = None,
                          access_expires_in: Optional[int] = None, refresh_expires_in: Optional[int] = None,
                          audience: Optional[str] = None, issuer: Optional[str] = None) -> Dict[str, str]:
        return {
            "access_token": self.create_access_token(subject, payload, access_expires_in, audience, issuer),
            "refresh_token": self.create_refresh_token(subject, payload, refresh_expires_in, audience, issuer),
            "token_type": "bearer",
        }

    def decode_token(self, token: str, verify_exp: bool = True, verify_aud: bool = True,
                     verify_iss: bool = True, expected_type: Optional[str] = None) -> Dict[str, Any]:
        opts = {"verify_exp": verify_exp, "verify_aud": verify_aud and bool(self.audience),
                "verify_iss": verify_iss and bool(self.issuer)}
        kwargs = {}
        if self.audience and verify_aud:
            kwargs["audience"] = self.audience
        if self.issuer and verify_iss:
            kwargs["issuer"] = self.issuer
        try:
            if self.algorithm.startswith("RS") and not self.secret_key:
                last = None
                for pem in self.public_keys:
                    try:
                        payload = jwt.decode(token, pem, algorithms=[self.algorithm], options=opts, **kwargs)
                        break
                    except Exception as e:
                        last = e
                else:
                    raise last or jwt.InvalidTokenError("RS256 verification failed")
            else:
                payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm], options=opts, **kwargs)
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
        except jwt.InvalidTokenError as e:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token: {e}")
        if expected_type and payload.get("type") != expected_type:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token type")
        return payload

    def refresh_access_token(self, refresh_token: str, payload: Optional[Dict] = None,
                             expires_in: Optional[int] = None, audience: Optional[str] = None,
                             issuer: Optional[str] = None) -> str:
        data = self.decode_token(refresh_token, expected_type="refresh")
        merged = {k: v for k, v in data.items() if k not in ("sub", "exp", "iat", "jti", "type", "aud", "iss")}
        if payload:
            merged.update(payload)
        return self.create_access_token(data["sub"], merged, expires_in, audience, issuer)

    def get_token_dependency(self, verify_exp: bool = True, verify_aud: bool = True,
                             verify_iss: bool = True, expected_type: Optional[str] = None) -> Callable:
        async def dependency(creds: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
            return self.decode_token(creds.credentials, verify_exp, verify_aud, verify_iss, expected_type)
        return dependency

    def get_user_dependency(self, verify_exp: bool = True, verify_aud: bool = True,
                            verify_iss: bool = True, expected_type: str = "access") -> Callable:
        token_dep = self.get_token_dependency(verify_exp, verify_aud, verify_iss, expected_type)
        async def dependency(payload: Dict[str, Any] = Depends(token_dep)) -> str:
            return payload["sub"]
        return dependency

    def get_scopes_dependency(self, required_scopes: List[str], verify_exp: bool = True,
                              verify_aud: bool = True, verify_iss: bool = True,
                              expected_type: str = "access") -> Callable:
        token_dep = self.get_token_dependency(verify_exp, verify_aud, verify_iss, expected_type)
        async def dependency(payload: Dict[str, Any] = Depends(token_dep)) -> Dict[str, Any]:
            missing = [s for s in required_scopes if s not in payload.get("scopes", [])]
            if missing:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Missing scopes: {missing}")
            return payload
        return dependency


jwt_auth = JWTAuth()


def require_auth(verify_exp: bool = True, verify_aud: bool = True,
                 verify_iss: bool = True, expected_type: Optional[str] = "access") -> Callable:
    return jwt_auth.get_token_dependency(verify_exp, verify_aud, verify_iss, expected_type)


def require_user(verify_exp: bool = True, verify_aud: bool = True,
                 verify_iss: bool = True, expected_type: str = "access") -> Callable:
    return jwt_auth.get_user_dependency(verify_exp, verify_aud, verify_iss, expected_type)


def require_scopes(required_scopes: List[str], verify_exp: bool = True, verify_aud: bool = True,
                   verify_iss: bool = True, expected_type: str = "access") -> Callable:
    return jwt_auth.get_scopes_dependency(required_scopes, verify_exp, verify_aud, verify_iss, expected_type)
