# cloud_decoder/dependencies.py
import os
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette import status

# Centralized secrets bootstrap (supports *_FILE envs)
from utils.secrets_bootstrap import bootstrap_secrets, get_secret

# Load file-based secrets into env, then read via centralized accessor
bootstrap_secrets(role="decoder")
INTERNAL_AUTH_TOKEN = get_secret("INTERNAL_SERVICE_TOKEN")
if not INTERNAL_AUTH_TOKEN:
    raise RuntimeError("INTERNAL_SERVICE_TOKEN not configured (env or *_FILE)")

api_key_header = APIKeyHeader(name="X-Internal-Auth", auto_error=False)

async def verify_internal_request(internal_token: str = Security(api_key_header)):
    """
    Dependency to verify that a request is coming from a trusted internal service.
    """
    if not internal_token or internal_token != INTERNAL_AUTH_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this internal endpoint.",
        )