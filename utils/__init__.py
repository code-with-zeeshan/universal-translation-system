# utils/__init__.py
"""
Common utilities for the Universal Translation System.

Provides shared utilities for directory management, logging,
and other common operations.
"""
from .common_utils import DirectoryManager
# Optional dependency: psutil (used by resource_tracker). Avoid hard requirement at import time.
try:
    from .resource_tracker import ResourceTracker, ResourceTracked, track_resources, resource_tracker  # type: ignore
except Exception:
    ResourceTracker = None  # type: ignore
    ResourceTracked = None  # type: ignore
    def track_resources(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func
        return decorator
    resource_tracker = None  # type: ignore
# Import base classes
from .base_classes import (
    BaseDataProcessor, BaseVocabularyManager, BaseVocabularyHandler,
    BaseManager, BaseProcessor, BaseValidator,
)

# Import thread safety utilities
from .thread_safety import (
    document_thread_safety, document_method_thread_safety,
    thread_safe, get_thread_safety_info, generate_thread_safety_report,
    THREAD_SAFETY_NONE, THREAD_SAFETY_EXTERNAL, THREAD_SAFETY_INTERNAL, THREAD_SAFETY_IMMUTABLE
)

# Import credential management
from .credential_manager import (
    CredentialManager, credential_manager, get_credential, set_credential,
)

# Import secure serialization
from .secure_serialization import (
    secure_serialize_json, secure_deserialize_json,
    secure_serialize_msgpack, secure_deserialize_msgpack,
    safe_deserialize_json,
)

# Import JWT authentication
# Optional dependency: pyjwt and cryptography (used by jwt_auth). Avoid hard requirement at import time.
try:
    from .jwt_auth import (
        JWTAuth, jwt_auth, require_auth, require_user,
    )  # type: ignore
except Exception:
    JWTAuth = None  # type: ignore
    def jwt_auth(*args, **kwargs):  # type: ignore
        raise ImportError("JWT features require 'pyjwt' and related deps installed")
    def require_auth(*args, **kwargs):  # type: ignore
        def decorator(func): return func
        return decorator
    def require_user(*args, **kwargs):  # type: ignore
        def decorator(func): return func
        return decorator
    def require_user(*args, **kwargs):  # type: ignore
        def decorator(func): return func
        return decorator

# Import exceptions
from .exceptions import (
    UniversalTranslationError, DataError, VocabularyError, ModelError,
    ConfigurationError, TrainingError, InferenceError, ResourceError,
    SecurityError, LoggingError, NetworkError,
)

__all__ = [
    "DirectoryManager",
    "document_thread_safety",
]
if ResourceTracker is not None:  # type: ignore
    __all__ += ["ResourceTracker", "ResourceTracked", "track_resources", "resource_tracker"]