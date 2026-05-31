import os
import stat
import time
import logging
from typing import Optional, Dict, Iterable, Tuple

from .credential_manager import credential_manager

logger = logging.getLogger(__name__)

ENV_TO_CRED_KEY: Dict[str, str] = {
    "COORDINATOR_SECRET": "coordinator_secret",
    "COORDINATOR_JWT_SECRET": "coordinator_jwt_secret",
    "COORDINATOR_TOKEN": "coordinator_token",
    "DECODER_JWT_SECRET": "decoder_jwt_secret",
    "INTERNAL_SERVICE_TOKEN": "internal_service_token",
    "JWT_SECRET": "jwt_secret",
    "JWT_PUBLIC_KEY": "jwt_public_key",
    "JWT_PUBLIC_KEY_PATH": "jwt_public_key_path",
    "JWT_PRIVATE_KEY": "jwt_private_key",
    "JWT_PRIVATE_KEY_FILE": "jwt_private_key_file",
    "UTS_HMAC_KEY": "uts_hmac_key",
    "REDIS_PASSWORD": "redis_password",
    "HF_TOKEN": "hf_token",
    "GRAFANA_ADMIN_PASSWORD": "grafana_admin_password",
}

FILE_ENV_PAIRS: Iterable[tuple[str, str]] = (
    ("COORDINATOR_SECRET_FILE", "COORDINATOR_SECRET"),
    ("COORDINATOR_JWT_SECRET_FILE", "COORDINATOR_JWT_SECRET"),
    ("COORDINATOR_TOKEN_FILE", "COORDINATOR_TOKEN"),
    ("INTERNAL_SERVICE_TOKEN_FILE", "INTERNAL_SERVICE_TOKEN"),
    ("DECODER_JWT_SECRET_FILE", "DECODER_JWT_SECRET"),
    ("JWT_PRIVATE_KEY_FILE", "JWT_PRIVATE_KEY"),
    ("REDIS_PASSWORD_FILE", "REDIS_PASSWORD"),
    ("HF_TOKEN_FILE", "HF_TOKEN"),
)

INSECURE_DEFAULTS = {
    "jwtsecret123",
    "changeme123",
    "a-very-secret-key-for-cookies",
    "a-super-secret-jwt-key",
    "internal-secret-token-for-service-auth",
    "use-openssl-rand-hex-32-to-generate-a-secure-key",
}


def _check_file_permissions(path: str) -> None:
    try:
        st = os.stat(path)
        if stat.S_IRWXG & st.st_mode or stat.S_IRWXO & st.st_mode:
            logger.warning("Secret file %s has group/world-accessible permissions; recommended: 0600", path)
    except OSError:
        pass


def _read_file_trim(path: str) -> Optional[str]:
    try:
        _check_file_permissions(path)
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.warning("Failed reading secret file %s: %s", path, e)
        return None


def bootstrap_secrets(role: Optional[str] = None) -> None:
    for file_env, target_env in FILE_ENV_PAIRS:
        file_path = os.environ.get(file_env)
        if file_path and not os.environ.get(target_env):
            content = _read_file_trim(file_path)
            if content:
                os.environ[target_env] = content
                logger.debug("Loaded secret for %s from %s (role=%s)", target_env, file_env, role)


def get_secret(env_name: str, default: Optional[str] = None) -> Optional[str]:
    key = ENV_TO_CRED_KEY.get(env_name, env_name.lower())
    value = credential_manager.get(key)
    if value:
        logger.debug("Secret %s resolved from credential manager", env_name)
        return value
    value = os.environ.get(env_name)
    if value:
        logger.debug("Secret %s resolved from environment", env_name)
        return value
    return default


def is_strong_secret(value: Optional[str], min_len: int = 32) -> bool:
    if not value or len(value) < min_len:
        return False
    if value in INSECURE_DEFAULTS:
        return False
    return True


# ---------------------------------------------------------------------------
# Secret expiry / rotation tracking
# ---------------------------------------------------------------------------
# To track expiry callers set a companion env var: <SECRET_NAME>_EXPIRY=<unix_ts>
# rotate_secret_if_expired() checks and rotates via credential_manager.

def get_secret_expiry(env_name: str) -> Optional[float]:
    expiry_str = os.environ.get(f"{env_name}_EXPIRY")
    if expiry_str:
        try:
            return float(expiry_str)
        except ValueError:
            logger.warning("Invalid expiry value for %s: %s", env_name, expiry_str)
    return None


def is_secret_expired(env_name: str) -> bool:
    expiry = get_secret_expiry(env_name)
    if expiry is None:
        return False
    return time.time() >= expiry


def rotate_secret_if_expired(env_name: str, rotate_func, min_len: int = 32) -> Tuple[bool, Optional[str]]:
    if not is_secret_expired(env_name):
        return False, None
    current = get_secret(env_name)
    if current and not is_strong_secret(current, min_len=min_len):
        logger.warning("Secret %s is weak, not rotating automatically", env_name)
        return False, None
    new_value = rotate_func()
    key = ENV_TO_CRED_KEY.get(env_name, env_name.lower())
    credential_manager.set(key, new_value)
    logger.info("Rotated expired secret %s", env_name)
    return True, new_value


def validate_runtime_secrets(role: Optional[str] = None) -> None:
    errors = []

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

        if priv:
            try:
                from cryptography.hazmat.primitives.serialization import load_pem_private_key
                key = load_pem_private_key(priv.encode("utf-8"), password=None)
                size = getattr(key, "key_size", None)
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
