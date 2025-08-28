# utils/__init__.py
"""
Common utilities for the Universal Translation System.

Provides shared utilities for directory management, logging,
and other common operations.
"""
from .dependency_container import container, inject, injectable
from .service_discovery import ServiceDiscoveryClient
from .common_utils import DirectoryManager,ImportCleaner
from .lazy_loader import LazyObject, lazy_property, lazy_function, LazyClass, lazy_import, lazy_singleton
from .cache_manager import Cache, CacheManager, cached, EvictionPolicy, cache_manager
from .batch_processor import BatchProcessor, AsyncBatchProcessor, batch_decorator, async_batch_decorator
from .resource_tracker import ResourceTracker, ResourceTracked, track_resources, resource_tracker
# Import constants
from .constants import *

# Import base classes
from .base_classes import (
    BaseDataProcessor, BaseVocabularyManager, BaseVocabularyHandler,
    BaseManager, BaseProcessor, BaseValidator, TokenizerMixin
)

# Import thread safety utilities
from .thread_safety import (
    document_thread_safety, document_method_thread_safety,
    thread_safe, get_thread_safety_info, generate_thread_safety_report,
    THREAD_SAFETY_NONE, THREAD_SAFETY_EXTERNAL, THREAD_SAFETY_INTERNAL, THREAD_SAFETY_IMMUTABLE
)

# Import validation utilities
from .validation_decorators import (
    validate_request_body, validate_query_params, validate_path_params,
    validate_input, validate_output, validate_model_input
)

# Import credential management
from .credential_manager import (
    CredentialManager, credential_manager, get_credential, set_credential, delete_credential
)

# Import secure serialization
from .secure_serialization import (
    secure_serialize_json, secure_deserialize_json,
    secure_serialize_msgpack, secure_deserialize_msgpack,
    safe_deserialize_json, safe_deserialize_msgpack
)

# Import JWT authentication
from .jwt_auth import (
    JWTAuth, jwt_auth, require_auth, require_user, require_scopes
)

# Import exceptions
from .exceptions import (
    UniversalTranslationError, DataError, VocabularyError, ModelError,
    ConfigurationError, TrainingError, InferenceError, ResourceError,
    SecurityError, LoggingError, NetworkError, TimeoutError
)

__all__ = [
    "DirectoryManager",
    "inject",
    "injectable",
    "container",
    "ImportCleaner",
    "ServiceDiscoveryClient",
    "LazyObject",
    "lazy_property",
    "lazy_function",
    "LazyClass",
    "lazy_import",
    "lazy_singleton",
    "Cache",
    "CacheManager",
    "cached",
    "EvictionPolicy",
    "cache_manager",
    "BatchProcessor",
    "AsyncBatchProcessor",
    "batch_decorator",
    "async_batch_decorator",
    "ResourceTracker",
    "ResourceTracked",
    "track_resources",
    "resource_tracker",
    "document_thread_safety",
]