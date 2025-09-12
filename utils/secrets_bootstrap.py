# utils/secrets_bootstrap.py
"""
Centralized secrets bootstrap and access helpers.
- Loads *_FILE envs into actual env vars at process start
- Provides get_secret() that prefers CredentialManager keys, with env fallbacks
- Validates minimum strength and RS256 key sizes where applicable
"""
from __future__ import annotations
import os
import logging
from typing import Optional, Dict, Iterable

from .credential_manager import credential_manager

logger = logging.getLogger(__name__)

# Map well-known environment variable names to credential keys
# This preserves compatibility while enabling a single access path
ENV_TO_CRED_KEY: Dict[str, str] = {
    # Coordinator
    "COORDINATOR_SECRET": "coordinator_secret",
    "COORDINATOR_JWT_SECRET": "coordinator_jwt_secret",
    "COORDINATOR_TOKEN": "coordinator_token",
    # Decoder
    "DECODER_JWT_SECRET": "decoder_jwt_secret",
    # Shared internal token
    "INTERNAL_SERVICE_TOKEN": "internal_service_token",
    # Generic JWT secret (for utils.jwt_auth default)
    "JWT_SECRET": "jwt_secret",
    # RS256 public/private
    "JWT_PUBLIC_KEY": "jwt_public_key",
    "JWT_PUBLIC_KEY_PATH": "jwt_public_key_path",
    "JWT_PRIVATE_KEY": "jwt_private_key",
    "JWT_PRIVATE_KEY_FILE": "jwt_private_key_file",
    # Secure serialization HMAC
    "UTS_HMAC_KEY": "uts_hmac_key",
}

# Pairs of (file env var, target env var)
FILE_ENV_PAIRS: Iterable[tuple[str, str]] = (
    ("COORDINATOR_SECRET_FILE", "COORDINATOR_SECRET"),
    ("COORDINATOR_JWT_SECRET_FILE", "COORDINATOR_JWT_SECRET"),
    ("COORDINATOR_TOKEN_FILE", "COORDINATOR_TOKEN"),
    ("INTERNAL_SERVICE_TOKEN_FILE", "INTERNAL_SERVICE_TOKEN"),
    ("DECODER_JWT_SECRET_FILE", "DECODER_JWT_SECRET"),
    ("JWT_PRIVATE_KEY_FILE", "JWT_PRIVATE_KEY"),  # keep PRIVATE in env for consumers that read it
)

INSECURE_DEFAULTS = {
    "jwtsecret123",
    "changeme123",
    "a-very-secret-key-for-cookies",
    "a-super-secret-jwt-key",
    "internal-secret-token-for-service-auth",
    "use-openssl-rand-hex-32-to-generate-a-secure-key",
}


def _read_file_trim(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.warning(f"Failed reading secret file {path}: {e}")
        return None


def bootstrap_secrets(role: Optional[str] = None) -> None:
    """
    Resolve *_FILE env vars to their target env vars.
    Idempotent: safe to call multiple times.
    role: optional "coordinator" | "decoder" to limit logging context
    """
    for file_env, target_env in FILE_ENV_PAIRS:
        file_path = os.environ.get(file_env)
        if file_path and not os.environ.get(target_env):
            content = _read_file_trim(file_path)
            if content:
                os.environ[target_env] = content
                logger.debug(f"Loaded secret for {target_env} from {file_env}")


def get_secret(env_name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get a secret with priority:
    1) CredentialManager (mapped key)
    2) Environment variable (env_name)
    3) default
    """
    key = ENV_TO_CRED_KEY.get(env_name, env_name.lower())
    # 1) Credential manager (UTS_ prefix applied inside)
    value = credential_manager.get(key)
    if value:
        return value
    # 2) Environment fallback
    return os.environ.get(env_name, default)


def is_strong_secret(value: Optional[str], min_len: int = 32) -> bool:
    if not value or len(value) < min_len:
        return False
    if value in INSECURE_DEFAULTS:
        return False
    return True


def validate_runtime_secrets(role: Optional[str] = None) -> None:
    """
    Validate presence and strength of required secrets for given role.
    - coordinator: COORDINATOR_SECRET, COORDINATOR_TOKEN, INTERNAL_SERVICE_TOKEN, and either COORDINATOR_JWT_SECRET (HS256) or RS256 keys
    - decoder: DECODER_JWT_SECRET and INTERNAL_SERVICE_TOKEN
    - always: UTS_HMAC_KEY should exist (secure_serialization enforces too)
    Raises RuntimeError with actionable messages on failure.
    """
    errors = []

    # Secure serialization HMAC key recommended everywhere
    if not is_strong_secret(os.environ.get("UTS_HMAC_KEY")):
        errors.append("UTS_HMAC_KEY must be set to a strong, random value (>=32 chars).")

    if role == "coordinator":
        if not is_strong_secret(get_secret("COORDINATOR_SECRET")):
            errors.append("COORDINATOR_SECRET must be strong (>=32 chars) and not a placeholder.")
        if not is_strong_secret(get_secret("COORDINATOR_TOKEN")):
            errors.append("COORDINATOR_TOKEN must be strong (>=32 chars) and not a placeholder.")
        if not is_strong_secret(get_secret("INTERNAL_SERVICE_TOKEN")):
            errors.append("INTERNAL_SERVICE_TOKEN must be strong (>=32 chars) and not a placeholder.")

        hs = get_secret("COORDINATOR_JWT_SECRET")
        pub = os.environ.get("JWT_PUBLIC_KEY") or os.environ.get("JWT_PUBLIC_KEY_PATH")
        priv = os.environ.get("JWT_PRIVATE_KEY")
        if not (is_strong_secret(hs) or (pub and priv)):
            errors.append("Provide either strong COORDINATOR_JWT_SECRET (HS256) or RS256 keys (JWT_PRIVATE_KEY and JWT_PUBLIC_KEY/PATH).")

        # If RS256 private key provided, validate size >= 2048 bits
        if priv:
            try:
                from cryptography.hazmat.primitives.serialization import load_pem_private_key
                from cryptography.hazmat.backends import default_backend as _db
                key = load_pem_private_key(priv.encode("utf-8"), password=None, backend=_db())
                size = getattr(getattr(key, "key_size", None), "__int__", lambda: key.key_size)()
                if hasattr(key, "key_size"):
                    size = key.key_size
                if not size or int(size) < 2048:
                    errors.append("RS256 private key must be >= 2048 bits.")
            except Exception as e:
                errors.append(f"Invalid RS256 private key: {e}")

    if role == "decoder":
        if not is_strong_secret(get_secret("DECODER_JWT_SECRET")):
            errors.append("DECODER_JWT_SECRET must be strong (>=32 chars) and not a placeholder.")
        if not is_strong_secret(get_secret("INTERNAL_SERVICE_TOKEN")):
            errors.append("INTERNAL_SERVICE_TOKEN must be strong (>=32 chars) and not a placeholder.")

    if errors:
        raise RuntimeError("Secret validation failed:\n- " + "\n- ".join(errors))