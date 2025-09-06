# utils/secure_serialization.py
"""
Secure serialization utilities for the Universal Translation System.
This module provides tools for securely serializing and deserializing data.
"""

import json
import msgpack
import pickle
import logging
import hashlib
import hmac
import os
import base64
import io
import builtins
from typing import Any, Dict, List, Optional, Union, Callable, Type, TypeVar
from .exceptions import SecurityError

logger = logging.getLogger(__name__)

# Default HMAC key for message authentication
DEFAULT_HMAC_KEY = os.environ.get("UTS_HMAC_KEY")
if not DEFAULT_HMAC_KEY or DEFAULT_HMAC_KEY == "change-me-in-production":
    raise SecurityError(
        "UTS_HMAC_KEY is not set or uses an insecure default. Set a strong random key via environment."
    )

# Maximum allowed size for deserialized data (100MB)
MAX_SIZE = 100 * 1024 * 1024

# Allowed types for deserialization
ALLOWED_TYPES = (
    type(None), bool, int, float, str, bytes, list, tuple, dict, set
)


def secure_serialize_json(data: Any, hmac_key: Optional[str] = None) -> str:
    """
    Securely serialize data to JSON with HMAC.
    
    Args:
        data: Data to serialize
        hmac_key: Key for HMAC (defaults to environment variable)
        
    Returns:
        Serialized data with HMAC
    """
    # Serialize data
    json_data = json.dumps(data)
    
    # Add HMAC
    key = hmac_key or DEFAULT_HMAC_KEY
    signature = hmac.new(
        key.encode(),
        json_data.encode(),
        hashlib.sha256
    ).hexdigest()
    
    # Return signed data
    return f"{signature}:{json_data}"


def secure_deserialize_json(data: str, hmac_key: Optional[str] = None) -> Any:
    """
    Securely deserialize JSON data with HMAC verification.
    
    Args:
        data: Serialized data with HMAC
        hmac_key: Key for HMAC (defaults to environment variable)
        
    Returns:
        Deserialized data
        
    Raises:
        SecurityError: If HMAC verification fails or data is invalid
    """
    # Check size
    if len(data) > MAX_SIZE:
        raise SecurityError(f"Data too large: {len(data)} bytes")
        
    # Split signature and data
    try:
        signature, json_data = data.split(':', 1)
    except ValueError:
        raise SecurityError("Invalid data format")
        
    # Verify HMAC
    key = hmac_key or DEFAULT_HMAC_KEY
    expected_signature = hmac.new(
        key.encode(),
        json_data.encode(),
        hashlib.sha256
    ).hexdigest()
    
    if not hmac.compare_digest(signature, expected_signature):
        raise SecurityError("HMAC verification failed")
        
    # Deserialize data
    try:
        return json.loads(json_data)
    except json.JSONDecodeError as e:
        raise SecurityError(f"Invalid JSON: {e}")


def secure_serialize_msgpack(data: Any, hmac_key: Optional[str] = None) -> bytes:
    """
    Securely serialize data to MessagePack with HMAC.
    
    Args:
        data: Data to serialize
        hmac_key: Key for HMAC (defaults to environment variable)
        
    Returns:
        Serialized data with HMAC
    """
    # Serialize data
    msgpack_data = msgpack.packb(data)
    
    # Add HMAC
    key = hmac_key or DEFAULT_HMAC_KEY
    signature = hmac.new(
        key.encode(),
        msgpack_data,
        hashlib.sha256
    ).digest()
    
    # Return signed data
    return signature + msgpack_data


def secure_deserialize_msgpack(data: bytes, hmac_key: Optional[str] = None) -> Any:
    """
    Securely deserialize MessagePack data with HMAC verification.
    
    Args:
        data: Serialized data with HMAC
        hmac_key: Key for HMAC (defaults to environment variable)
        
    Returns:
        Deserialized data
        
    Raises:
        SecurityError: If HMAC verification fails or data is invalid
    """
    # Check size
    if len(data) > MAX_SIZE:
        raise SecurityError(f"Data too large: {len(data)} bytes")
        
    # Split signature and data
    signature = data[:32]  # SHA-256 is 32 bytes
    msgpack_data = data[32:]
    
    # Verify HMAC
    key = hmac_key or DEFAULT_HMAC_KEY
    expected_signature = hmac.new(
        key.encode(),
        msgpack_data,
        hashlib.sha256
    ).digest()
    
    if not hmac.compare_digest(signature, expected_signature):
        raise SecurityError("HMAC verification failed")
        
    # Deserialize data
    try:
        return msgpack.unpackb(msgpack_data)
    except Exception as e:
        raise SecurityError(f"Invalid MessagePack: {e}")


def validate_type(obj: Any) -> None:
    """
    Validate that an object only contains allowed types.
    
    Args:
        obj: Object to validate
        
    Raises:
        SecurityError: If the object contains disallowed types
    """
    if not isinstance(obj, ALLOWED_TYPES):
        raise SecurityError(f"Disallowed type: {type(obj).__name__}")
        
    if isinstance(obj, (list, tuple)):
        for item in obj:
            validate_type(item)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(key, (str, int, float, bool)):
                raise SecurityError(f"Disallowed key type: {type(key).__name__}")
            validate_type(value)
    elif isinstance(obj, set):
        for item in obj:
            validate_type(item)


def safe_deserialize_json(data: str) -> Any:
    """
    Safely deserialize JSON data with type validation.
    
    Args:
        data: JSON string
        
    Returns:
        Deserialized data
        
    Raises:
        SecurityError: If the data contains disallowed types
    """
    # Check size
    if len(data) > MAX_SIZE:
        raise SecurityError(f"Data too large: {len(data)} bytes")
        
    # Deserialize data
    try:
        obj = json.loads(data)
    except json.JSONDecodeError as e:
        raise SecurityError(f"Invalid JSON: {e}")
        
    # Validate types
    validate_type(obj)
    
    return obj


def safe_deserialize_msgpack(data: bytes) -> Any:
    """
    Safely deserialize MessagePack data with type validation.
    
    Args:
        data: MessagePack bytes
        
    Returns:
        Deserialized data
        
    Raises:
        SecurityError: If the data contains disallowed types
    """
    # Check size
    if len(data) > MAX_SIZE:
        raise SecurityError(f"Data too large: {len(data)} bytes")
        
    # Deserialize data
    try:
        obj = msgpack.unpackb(data)
    except Exception as e:
        raise SecurityError(f"Invalid MessagePack: {e}")
        
    # Validate types
    validate_type(obj)
    
    return obj


def safe_deserialize_pickle(data: bytes, allowed_modules: Optional[List[str]] = None) -> Any:
    """
    Safely deserialize pickle data with restricted globals.
    
    WARNING: Pickle deserialization is inherently unsafe. Use only with trusted data.
    
    Args:
        data: Pickle bytes
        allowed_modules: List of allowed modules
        
    Returns:
        Deserialized data
        
    Raises:
        SecurityError: If deserialization fails or uses disallowed modules
    """
    # Check size
    if len(data) > MAX_SIZE:
        raise SecurityError(f"Data too large: {len(data)} bytes")
        
    # Create restricted unpickler
    class RestrictedUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Check if module is allowed
            if allowed_modules and module not in allowed_modules:
                raise SecurityError(f"Disallowed module: {module}")
                
            # Only allow safe builtins
            if module == "builtins" and name in (
                "range", "complex", "set", "frozenset", "slice"
            ):
                return getattr(builtins, name)
                
            # Disallow all other imports
            raise SecurityError(f"Disallowed import: {module}.{name}")
            
    # Deserialize data
    try:
        return RestrictedUnpickler(io.BytesIO(data)).load()
    except Exception as e:
        raise SecurityError(f"Pickle deserialization failed: {e}")

def secure_serialize_json_compressed(data: Any, hmac_key: Optional[str] = None) -> str:
    """
    Securely serialize and compress data to JSON with HMAC.
    
    Args:
        data: Data to serialize
        hmac_key: Key for HMAC
        
    Returns:
        Compressed serialized data with HMAC
    """
    import zlib
    
    # Serialize data
    json_data = json.dumps(data)
    
    # Compress
    compressed = zlib.compress(json_data.encode())
    
    # Encode as base64
    b64_data = base64.b64encode(compressed).decode()
    
    # Add HMAC
    key = hmac_key or DEFAULT_HMAC_KEY
    signature = hmac.new(
        key.encode(),
        b64_data.encode(),
        hashlib.sha256
    ).hexdigest()
    
    # Return signed data
    return f"{signature}:{b64_data}"
    
def secure_deserialize_json_compressed(data: str, hmac_key: Optional[str] = None) -> Any:
    """
    Securely deserialize compressed JSON data with HMAC verification.
    
    Args:
        data: Compressed serialized data with HMAC
        hmac_key: Key for HMAC
        
    Returns:
        Deserialized data
        
    Raises:
        SecurityError: If HMAC verification fails or data is invalid
    """
    import zlib
    
    
    # Check size
    if len(data) > MAX_SIZE:
        raise SecurityError(f"Data too large: {len(data)} bytes")
        
    # Split signature and data
    try:
        signature, b64_data = data.split(':', 1)
    except ValueError:
        raise SecurityError("Invalid data format")
        
    # Verify HMAC
    key = hmac_key or DEFAULT_HMAC_KEY
    expected_signature = hmac.new(
        key.encode(),
        b64_data.encode(),
        hashlib.sha256
    ).hexdigest()
    
    if not hmac.compare_digest(signature, expected_signature):
        raise SecurityError("HMAC verification failed")
        
    try:
        # Decode base64
        compressed = base64.b64decode(b64_data)
        
        # Decompress
        json_data = zlib.decompress(compressed).decode()
        
        # Deserialize
        return json.loads(json_data)
    except Exception as e:
        raise SecurityError(f"Deserialization failed: {e}")

def secure_serialize_with_version(data: Any, version: int = 1, 
                                hmac_key: Optional[str] = None) -> str:
    """
    Securely serialize data with version information.
    
    Args:
        data: Data to serialize
        version: Data format version
        hmac_key: Key for HMAC
        
    Returns:
        Serialized data with version and HMAC
    """
    # Add version to data
    versioned_data = {
        "version": version,
        "data": data
    }
    
    # Serialize with HMAC
    return secure_serialize_json(versioned_data, hmac_key)
    
def secure_deserialize_with_version(data: str, 
                                  supported_versions: List[int] = [1],
                                  hmac_key: Optional[str] = None) -> Any:
    """
    Securely deserialize versioned data.
    
    Args:
        data: Serialized data with version and HMAC
        supported_versions: List of supported versions
        hmac_key: Key for HMAC
        
    Returns:
        Deserialized data
        
    Raises:
        SecurityError: If HMAC verification fails or version is not supported
    """
    # Deserialize with HMAC verification
    versioned_data = secure_deserialize_json(data, hmac_key)
    
    # Check version
    if not isinstance(versioned_data, dict) or "version" not in versioned_data:
        raise SecurityError("Missing version information")
        
    version = versioned_data.get("version")
    if version not in supported_versions:
        raise SecurityError(f"Unsupported version: {version}")
        
    # Return data
    return versioned_data.get("data")

def secure_deserialize_with_schema(data: str, schema: Dict[str, Any], 
                                 hmac_key: Optional[str] = None) -> Any:
    """
    Securely deserialize JSON data with schema validation.
    
    Args:
        data: Serialized data with HMAC
        schema: JSON Schema for validation
        hmac_key: Key for HMAC
        
    Returns:
        Validated deserialized data
        
    Raises:
        SecurityError: If HMAC verification fails or schema validation fails
    """
    # Deserialize with HMAC verification
    obj = secure_deserialize_json(data, hmac_key)
    
    # Validate schema
    try:
        import jsonschema
        jsonschema.validate(obj, schema)
        return obj
    except jsonschema.exceptions.ValidationError as e:
        raise SecurityError(f"Schema validation failed: {e}")
