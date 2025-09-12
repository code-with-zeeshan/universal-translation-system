# utils/constants.py
"""
Centralized constants for the Universal Translation System with environment overrides.

All values here can be overridden via environment variables. We check both
UTS_<NAME> and <NAME> to make it easy to integrate with different env setups.

Example:
- UTS_DEFAULT_TIMEOUT=45
- DEFAULT_TIMEOUT=45

Parsing rules:
- int/float are cast safely; on failure the default is kept
- bool accepts true/1/yes/on (case-insensitive) as True, false/0/no/off as False
- list accepts either a JSON array (e.g. ["<pad>","<unk>"]) or a comma-separated list
"""

from __future__ import annotations
import os
import json
from typing import Any, List


def _get_env_raw(key: str, default: Any = None) -> Any:
    """Get raw env value, checking UTS_<KEY> first, then <KEY>."""
    return os.environ.get(f"UTS_{key}", os.environ.get(key, default))


def _as_str(key: str, default: str) -> str:
    val = _get_env_raw(key)
    return default if val is None else str(val)


def _as_int(key: str, default: int) -> int:
    val = _get_env_raw(key)
    if val is None:
        return default
    try:
        return int(str(val).strip())
    except Exception:
        return default


def _as_float(key: str, default: float) -> float:
    val = _get_env_raw(key)
    if val is None:
        return default
    try:
        return float(str(val).strip())
    except Exception:
        return default


def _as_bool(key: str, default: bool) -> bool:
    val = _get_env_raw(key)
    if val is None:
        return default
    s = str(val).strip().lower()
    if s in {"true", "1", "yes", "on"}:
        return True
    if s in {"false", "0", "no", "off"}:
        return False
    return default


def _as_list(key: str, default: List[str]) -> List[str]:
    val = _get_env_raw(key)
    if val is None:
        return default
    s = str(val).strip()
    # Try JSON array first
    if s.startswith("["):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            return default
    # Fallback: comma-separated list
    if s == "":
        return default
    return [item.strip() for item in s.split(",") if item.strip()]


# General constants
DEFAULT_ENCODING = _as_str("DEFAULT_ENCODING", "utf-8")
DEFAULT_TIMEOUT = _as_int("DEFAULT_TIMEOUT", 30)  # seconds
MAX_RETRY_COUNT = _as_int("MAX_RETRY_COUNT", 3)
DEFAULT_BATCH_SIZE = _as_int("DEFAULT_BATCH_SIZE", 64)
DEFAULT_BUFFER_SIZE = _as_int("DEFAULT_BUFFER_SIZE", 8192)  # bytes

# Memory and resource limits
MAX_CACHE_SIZE = _as_int("MAX_CACHE_SIZE", 10000)  # items
DEFAULT_CACHE_TTL = _as_int("DEFAULT_CACHE_TTL", 3600)  # seconds
MAX_MEMORY_USAGE = _as_int("MAX_MEMORY_USAGE", 1024 * 1024 * 1024)  # 1GB
MAX_FILE_SIZE = _as_int("MAX_FILE_SIZE", 100 * 1024 * 1024)  # 100MB

# Security constants
TOKEN_EXPIRATION = _as_int("TOKEN_EXPIRATION", 30 * 60)  # 30 minutes
REFRESH_TOKEN_EXPIRATION = _as_int("REFRESH_TOKEN_EXPIRATION", 7 * 24 * 60 * 60)  # 7 days
PASSWORD_MIN_LENGTH = _as_int("PASSWORD_MIN_LENGTH", 8)
PASSWORD_MAX_LENGTH = _as_int("PASSWORD_MAX_LENGTH", 128)
MAX_LOGIN_ATTEMPTS = _as_int("MAX_LOGIN_ATTEMPTS", 5)
LOCKOUT_DURATION = _as_int("LOCKOUT_DURATION", 15 * 60)  # 15 minutes

# Encoder constants
ENCODER_EMBEDDING_DIM = _as_int("ENCODER_EMBEDDING_DIM", 512)
ENCODER_HIDDEN_DIM = _as_int("ENCODER_HIDDEN_DIM", 1024)
ENCODER_NUM_LAYERS = _as_int("ENCODER_NUM_LAYERS", 6)
ENCODER_NUM_HEADS = _as_int("ENCODER_NUM_HEADS", 8)
ENCODER_DROPOUT = _as_float("ENCODER_DROPOUT", 0.1)
ENCODER_MAX_LENGTH = _as_int("ENCODER_MAX_LENGTH", 512)

# Decoder constants
DECODER_EMBEDDING_DIM = _as_int("DECODER_EMBEDDING_DIM", 512)
DECODER_HIDDEN_DIM = _as_int("DECODER_HIDDEN_DIM", 1024)
DECODER_NUM_LAYERS = _as_int("DECODER_NUM_LAYERS", 6)
DECODER_NUM_HEADS = _as_int("DECODER_NUM_HEADS", 8)
DECODER_DROPOUT = _as_float("DECODER_DROPOUT", 0.1)
DECODER_MAX_LENGTH = _as_int("DECODER_MAX_LENGTH", 512)

# Vocabulary constants
VOCAB_SIZE = _as_int("VOCAB_SIZE", 32000)
VOCAB_MIN_FREQUENCY = _as_int("VOCAB_MIN_FREQUENCY", 5)
VOCAB_SPECIAL_TOKENS = _as_list("VOCAB_SPECIAL_TOKENS", ["<pad>", "<unk>", "<bos>", "<eos>"])
VOCAB_PAD_ID = _as_int("VOCAB_PAD_ID", 0)
VOCAB_UNK_ID = _as_int("VOCAB_UNK_ID", 1)
VOCAB_BOS_ID = _as_int("VOCAB_BOS_ID", 2)
VOCAB_EOS_ID = _as_int("VOCAB_EOS_ID", 3)

# API constants
API_RATE_LIMIT = _as_int("API_RATE_LIMIT", 100)  # requests per minute
API_BURST_LIMIT = _as_int("API_BURST_LIMIT", 20)  # concurrent requests
API_TIMEOUT = _as_int("API_TIMEOUT", 30)  # seconds
# Shared API version used across encoder/decoder/coordinator
API_VERSION = _as_str("API_VERSION", "1.0.0")
# Supported vocabulary manifest major format (e.g., '1' allows 1.x)
SUPPORTED_VOCAB_FORMAT = _as_str("SUPPORTED_VOCAB_FORMAT", "1")

# File paths
DEFAULT_CONFIG_PATH = _as_str("DEFAULT_CONFIG_PATH", "config/default_config.json")
DEFAULT_MODEL_PATH = _as_str("DEFAULT_MODEL_PATH", "models/default_model")
DEFAULT_VOCAB_PATH = _as_str("DEFAULT_VOCAB_PATH", "vocabulary/default_vocab")
DEFAULT_LOG_PATH = _as_str("DEFAULT_LOG_PATH", "logs/system.log")

# HTTP status codes (fixed, not expected to override)
HTTP_OK = 200
HTTP_CREATED = 201
HTTP_ACCEPTED = 202
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404
HTTP_SERVER_ERROR = 500

# Error messages (fixed)
ERROR_INVALID_INPUT = "Invalid input provided"
ERROR_UNAUTHORIZED = "Unauthorized access"
ERROR_RESOURCE_NOT_FOUND = "Resource not found"
ERROR_INTERNAL_SERVER = "Internal server error"
ERROR_RATE_LIMITED = "Rate limit exceeded"
ERROR_INVALID_TOKEN = "Invalid or expired token"