# cloud_decoder/dependencies.py
import os
import threading
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette import status

# Centralized secrets bootstrap (supports *_FILE envs)
from utils.secrets_bootstrap import bootstrap_secrets, get_secret

# Lazy initialization to avoid crashes during training imports
_internal_auth_token = None
_lock = threading.Lock()

def _get_internal_token() -> str:
    global _internal_auth_token
    if _internal_auth_token is None:
        with _lock:
            if _internal_auth_token is None:
                bootstrap_secrets(role="decoder")
                token = get_secret("INTERNAL_SERVICE_TOKEN")
                if not token:
                    raise RuntimeError("INTERNAL_SERVICE_TOKEN not configured (env or *_FILE)")
                _internal_auth_token = token
    return _internal_auth_token

api_key_header = APIKeyHeader(name="X-Internal-Auth", auto_error=False)

async def verify_internal_request(internal_token: str = Security(api_key_header)):
    """
    Dependency to verify that a request is coming from a trusted internal service.
    """
    token = _get_internal_token()
    if not internal_token or internal_token != token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this internal endpoint.",
        )