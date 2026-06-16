# cloud_decoder/optimized_decoder.py
import torch
import torch.nn as nn
from pathlib import Path
from fastapi import FastAPI, Request, Header, Depends, HTTPException, APIRouter
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from fastapi.responses import JSONResponse

# Crypto for JWKS and KID computation (mirrored)
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
import base64
import asyncio
from typing import List, Dict, Optional, Tuple, Any, Union
import time
# Optional: available in NVIDIA Triton Python backend environments
try:
    import triton_python_backend_utils as pb_utils  # type: ignore
except Exception:
    pb_utils = None  # Safe fallback when not running under Triton
import logging
import os
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from monitoring.metrics_collector import (
    track_translation_request,
    collect_vocabulary_metrics,
    get_metrics_summary,
    active_connections,
)
# Shared constants for API and vocab format enforcement
from utils.constants import API_VERSION, SUPPORTED_VOCAB_FORMAT, VERSION_CONFIG_FILENAME
from utils.logging_config import setup_logging, get_logger
from collections import OrderedDict
# Import core model classes (no server deps)
from .decoder_core import OptimizedUniversalDecoder, OptimizedDecoderLayer, ContinuousBatcher, decompress_encoder_output, decode_tokens_to_text
from utils.common_utils import RuntimeDirectoryManager

# Import vocabulary manager
from runtime.vocabulary.manager import UnifiedVocabularyManager, VocabularyMode

# Use OPTIMIZED mode for cloud deployment
VocabularyManager = lambda *args, **kwargs: UnifiedVocabularyManager(*args, mode=VocabularyMode.OPTIMIZED, **kwargs)

# --- ADDED: Hugging Face Hub Integration ---
try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    hf_hub_download = None

# Import utility modules
from utils.auth import APIKeyManager
from utils.rate_limiter import RateLimiter
from utils.security import validate_model_source, safe_load_model
from runtime.encoder.language_adapters import AdapterUniversalEncoder
# +++ ADDED: Import the new dependency +++
from .dependencies import verify_internal_request

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# OpenTelemetry setup
trace.set_tracer_provider(
    TracerProvider(resource=Resource.create({SERVICE_NAME: "cloud-decoder"}))
)
otlp_exporter = OTLPSpanExporter()
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)
tracer = trace.get_tracer(__name__)

# Environment variables for configuration
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.0.0")
# API version is imported from utils.constants (env-overridable)

# Centralized secrets bootstrap and access (only in decoder service context)
from utils.secrets_bootstrap import bootstrap_secrets, get_secret, validate_runtime_secrets
try:
    bootstrap_secrets(role="decoder")
    validate_runtime_secrets(role="decoder")
except Exception:
    logger.debug("secrets_bootstrap_failed", exc_info=True)

# Optional: prefetch artifacts on startup from HF Hub based on env hints
try:
    from utils.artifact_store import ArtifactStore
    store = ArtifactStore()
    packs = os.environ.get("PREFETCH_VOCAB_GROUPS", "").split(",") if os.environ.get("PREFETCH_VOCAB_GROUPS") else []
    adapters = os.environ.get("PREFETCH_ADAPTERS", "").split(",") if os.environ.get("PREFETCH_ADAPTERS") else []
    models = os.environ.get("PREFETCH_MODELS", "").split(",") if os.environ.get("PREFETCH_MODELS") else []
    for p in filter(None, [s.strip() for s in packs]):
        try:
            store.ensure_vocab_pack(p)
        except Exception as e:
            logger.warning(f"Prefetch pack failed for {p}: {e}")
    for a in filter(None, [s.strip() for s in adapters]):
        try:
            store.ensure_adapter(a)
        except Exception as e:
            logger.warning(f"Prefetch adapter failed for {a}: {e}")
    for m in filter(None, [s.strip() for s in models]):
        try:
            store.ensure_model(m)
        except Exception as e:
            logger.warning(
                "prefetch_model_failed",
                extra={"model": m, "error": str(e)}
            )
except Exception as e:
    logger.info(
        "artifact_prefetch_skipped",
        extra={"error": str(e)}
    )
JWT_SECRET = get_secret("DECODER_JWT_SECRET") if os.environ.get("DECODER_JWT_SECRET") or os.environ.get("DECODER_JWT_SECRET_FILE") else None
CONFIG_PATH = os.environ.get("DECODER_CONFIG_PATH", "config/decoder_config.yaml")
HF_HUB_REPO_ID = os.environ.get("HF_HUB_REPO_ID", "your-hf-org/universal-translation-system")

# API endpoints and server configuration
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))
API_WORKERS = int(os.environ.get("API_WORKERS", "1"))
API_TITLE = os.environ.get("API_TITLE", "Cloud Decoder API")

# Initialize logging (centralized)
setup_logging(log_dir=str(RuntimeDirectoryManager().logs_dir), log_level=os.environ.get("LOG_LEVEL", "INFO"))
# Use structured logger adapter to encourage structured fields usage
logger = get_logger("decoder", context={"component": "decoder"})

# Initialize utilities
api_key_manager = APIKeyManager()
rate_limiter = RateLimiter(requests_per_minute=60, requests_per_hour=1000)

# AdapterManager Class 
class AdapterManager:
    """
    Manages dynamic loading and LRU caching of language adapters on a decoder node.
    """
    def __init__(self, model: nn.Module, repo_id: str, max_cache_size: int = 5, adapter_dir: Optional[str] = None):
        self.model = model  # The AdapterUniversalEncoder instance
        self.adapter_dir = Path(adapter_dir) if adapter_dir is not None else RuntimeDirectoryManager().adapters_dir
        self.max_cache_size = max_cache_size
        self.repo_id = repo_id
        
        # Use an OrderedDict for a simple and effective LRU cache
        # Maps adapter_name -> loaded adapter module
        self.cache = OrderedDict()
        # --- MODIFIED: Use asyncio.Lock for async environments ---
        self.lock = asyncio.Lock()
        self.loading_events: Dict[str, asyncio.Event] = {}
        
        logger.info(
            "adapter_manager_initialized",
            extra={"repo": self.repo_id, "cache_size": self.max_cache_size}
        )

    async def get_adapter(self, adapter_name: str):
        """
        Asynchronously retrieves a language adapter, loading it from disk if not in the cache.
        This is now non-blocking for high-concurrency environments.
        """
        # --- Non-blocking cache check ---
        async with self.lock:
            if adapter_name in self.cache: # Cache Hit
                self.cache.move_to_end(adapter_name) # Mark as recently used
                logger.debug(
                    "adapter_cache_hit",
                    extra={"adapter": adapter_name}
                )
                return self.cache[adapter_name]

            # --- Cache Miss ---
            # Check if another thread is already loading this adapter
            if adapter_name in self.loading_events:
                # Another coroutine is loading this adapter; wait for it without holding the lock
                event = self.loading_events[adapter_name]
            else:
                event = None

        # If someone else is loading, wait here (lock released)
        if event is not None:
            logger.info(
                "adapter_loading_wait",
                extra={"adapter": adapter_name}
            )
            await event.wait()
            # After waiting, check cache again
            async with self.lock:
                if adapter_name in self.cache:
                    self.cache.move_to_end(adapter_name)
                    return self.cache[adapter_name]
                else:
                    logger.error(
                        "adapter_wait_failed",
                        extra={"adapter": adapter_name}
                    )
                    raise RuntimeError(f"Failed to load adapter {adapter_name}")

            # This is the first thread to request this adapter, so it will load it.
            # Create an event to signal other threads (inside the lock to prevent TOCTOU).
            async with self.lock:
                if adapter_name in self.cache:
                    self.cache.move_to_end(adapter_name)
                    return self.cache[adapter_name]
                if adapter_name not in self.loading_events:
                    self.loading_events[adapter_name] = asyncio.Event()

        # --- Perform slow I/O outside the main lock ---
        try:
            # Download the adapter from the Hub if it doesn't exist locally
            await self._download_adapter_from_hub(adapter_name)

            # --- Re-acquire lock to update the shared cache state ---
            async with self.lock:
                logger.info(
                    "adapter_cache_miss_loading",
                    extra={"adapter": adapter_name}
                )
                # Evict if cache is full
                if len(self.cache) >= self.max_cache_size:
                    oldest_adapter_name, _ = self.cache.popitem(last=False)
                    if oldest_adapter_name in self.model.language_adapters:
                        del self.model.language_adapters[oldest_adapter_name]
                    logger.info(
                        "adapter_cache_evicted",
                        extra={"adapter": oldest_adapter_name}
                    )

                # Load the new adapter into the model
                adapter_path = self.adapter_dir / f"best_{adapter_name}_adapter.pt"
                self.model.load_language_adapter(adapter_name, str(adapter_path))
                self.cache[adapter_name] = self.model.language_adapters[adapter_name]
                logger.info(
                    "adapter_cache_loaded",
                    extra={"adapter": adapter_name}
                )
                # Return the loaded adapter instance
                return self.cache[adapter_name]
        finally:
            # --- Signal other waiting threads and clean up ---
            async with self.lock:
                if adapter_name in self.loading_events:
                    self.loading_events[adapter_name].set()
                    del self.loading_events[adapter_name]

    async def _download_adapter_from_hub(self, adapter_name: str):
        """
        Downloads an adapter from the Hugging Face Hub if it doesn't exist locally.
        This method is non-blocking.
        """
        if not hf_hub_download:
            raise ImportError("huggingface_hub is not installed. Please install it with 'pip install huggingface_hub'")

        adapter_filename = f"best_{adapter_name}_adapter.pt"
        local_adapter_path = self.adapter_dir / adapter_filename
        
        if local_adapter_path.exists():
            logger.debug(
                "adapter_exists_locally",
                extra={"adapter": adapter_name, "path": str(local_adapter_path)}
            )
            return

        logger.info(
            "adapter_download_begin",
            extra={"adapter": adapter_name, "repo": self.repo_id}
        )
        
        # The filename within the HF repo (e.g., "adapters/best_es_adapter.pt")
        repo_filename = f"adapters/{adapter_filename}"
        
        loop = asyncio.get_event_loop()
        try:
            # Run the synchronous hf_hub_download in a thread pool executor to avoid blocking the event loop.
            await loop.run_in_executor(
                None,  # Use default executor
                hf_hub_download,
                self.repo_id,
                repo_filename,
                local_dir=str(self.adapter_dir),
                local_dir_use_symlinks=False  # Use copies for robustness in containers
            )
            logger.info(
                "adapter_download_success",
                extra={"adapter": adapter_name, "path": str(local_adapter_path)}
            )
        except HfHubHTTPError as e:
            if e.response.status_code == 404:
                logger.error(
                    "adapter_download_404",
                    extra={"repo_file": repo_filename, "repo": self.repo_id}
                )
                raise FileNotFoundError(f"Adapter not found on Hub: {repo_filename}") from e
            logger.error(
                "adapter_download_http_error",
                extra={"adapter": adapter_name, "error": str(e)}
            )
            raise IOError(f"Failed to download adapter: {e}") from e
        except Exception as e:
            logger.error(
                "adapter_download_unexpected_error",
                extra={"adapter": adapter_name, "error": str(e)}
            )
            raise IOError(f"Failed to download adapter: {e}") from e

    async def get_loaded_adapters(self) -> List[str]:
        """Returns a list of adapters currently hot in the cache."""
        async with self.lock:
            return list(self.cache.keys())

    


# Pydantic model for the request body
class CompositionRequest(BaseModel):
    source_adapter: str
    target_adapter: str
    strategy: str = "average"

# FastAPI application for serving
# Use shared API_VERSION for API surface; keep MODEL_VERSION in /status
app = FastAPI(title=API_TITLE, version=API_VERSION, openapi_url="/openapi.json")
FastAPIInstrumentor.instrument_app(app)

@app.on_event("startup")
async def startup_validation():
    # Centralized validation for decoder secrets and policy
    try:
        validate_runtime_secrets(role="decoder")
    except Exception as ex:
        logger.error(f"Decoder secret validation failed: {ex}")
        raise

    # Policy: API_VERSION must match core.apiVersion; vocab format must be supported
    try:
        import json
        cfg = json.loads(Path(VERSION_CONFIG_FILENAME).read_text(encoding="utf-8"))
        core_api = str(cfg.get("core", {}).get("apiVersion", ""))
        if not core_api:
            raise RuntimeError("core.apiVersion missing in version-config.json")
        if str(API_VERSION) != core_api:
            raise RuntimeError(f"API_VERSION ({API_VERSION}) != core.apiVersion ({core_api})")
        manifest = RuntimeDirectoryManager().vocab_manifest_path
        if manifest.exists():
            data = json.loads(manifest.read_text(encoding="utf-8"))
            fmt = str(data.get("format_version", "")).split(".")[0]
            if fmt and fmt != str(SUPPORTED_VOCAB_FORMAT).split(".")[0]:
                raise RuntimeError(
                    f"Unsupported vocabulary format {data.get('format_version')} (supported major: {SUPPORTED_VOCAB_FORMAT})"
                )
    except Exception as ex:
        logger.error(f"Startup policy check failed: {ex}")
        raise

    # Build JWKS using shared utility
    from utils.jwks_utils import build_jwks_from_env, diff_kids
    jwks_keys = build_jwks_from_env(component="decoder", env=os.environ)
    set_jwks_keys(jwks_keys)

    # Background job to periodically reload JWKS every 5 minutes, with change logging
    async def jwks_reload_loop_decoder():
        last_keys = get_jwks_keys()
        while True:
            try:
                new_keys = build_jwks_from_env(component="decoder", env=os.environ)
                if new_keys:
                    added, removed = diff_kids(last_keys, new_keys)
                    set_jwks_keys(new_keys)
                    if added or removed:
                        logger.info(f"Decoder JWKS changed. Added: {added or '[]'}, Removed: {removed or '[]'}")
                    last_keys = new_keys
            except Exception as ex:
                logger.debug(f"Decoder JWKS reload loop error: {ex}")
            await asyncio.sleep(300)

    global JWKS_RELOAD_TASK
    JWKS_RELOAD_TASK = asyncio.create_task(jwks_reload_loop_decoder())

# CORS and Security headers middleware
from fastapi.middleware.cors import CORSMiddleware

# Wire graceful shutdown
from utils.shutdown_handler import GracefulShutdown

shutdown_ref = None
# Track background task and file observer for graceful shutdown
JWKS_RELOAD_TASK: Optional[asyncio.Task] = None
FILE_OBSERVER: Optional[Observer] = None

@app.on_event("startup")
async def _install_shutdown_handler():
    """Install graceful shutdown to cleanup resources."""
    global shutdown_ref
    def cleanup():
        try:
            try:
                if JWKS_RELOAD_TASK:
                    JWKS_RELOAD_TASK.cancel()
            except Exception:
                logger.warning("jwks_reload_task_cancel_failed", exc_info=True)
            try:
                if FILE_OBSERVER:
                    FILE_OBSERVER.stop()
                    FILE_OBSERVER.join(timeout=1.0)
            except Exception:
                logger.warning("file_observer_stop_failed", exc_info=True)
            try:
                from utils.redis_manager import RedisManager
                RedisManager.get_instance().stop_health_check()
            except Exception:
                logger.warning("redis_health_check_stop_failed", exc_info=True)
            try:
                provider = trace.get_tracer_provider()
                shutdown = getattr(provider, "shutdown", None)
                if callable(shutdown):
                    shutdown()
            except Exception:
                logger.warning("otel_shutdown_failed", exc_info=True)
        except Exception:
            logger.warning("cleanup_all_failed", exc_info=True)
    shutdown_ref = GracefulShutdown(cleanup)

# Restrictive defaults; adjust ALLOWED_ORIGINS via env if needed
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "").split(",") if os.environ.get("ALLOWED_ORIGINS") else []
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Client-ID", "X-Request-ID"],
)

@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'; base-uri 'none'"
    return response

security = HTTPBearer()
JWT_ISS = os.environ.get("JWT_ISS")
JWT_AUD = os.environ.get("JWT_AUD")
JWT_PUBLIC_KEY = os.environ.get("JWT_PUBLIC_KEY")
JWT_PUBLIC_KEY_PATH = os.environ.get("JWT_PUBLIC_KEY_PATH")
JWT_KEY_IDS = os.environ.get("JWT_KEY_IDS")
JWKS_KEYS: List[Dict[str, Any]] = []


def require_jwt(credentials: HTTPAuthorizationCredentials = Depends(security)):
    import jwt
    token = credentials.credentials
    try:
        if JWT_PUBLIC_KEY or JWT_PUBLIC_KEY_PATH:
            # Select key by kid when present
            unverified_header = jwt.get_unverified_header(token)
            kid = unverified_header.get("kid")
            key_to_use = None
            jwks_keys = get_jwks_keys()
            if kid and jwks_keys:
                for k in jwks_keys:
                    if k.get("kid") == kid:
                        n = int.from_bytes(base64.urlsafe_b64decode(k["n"] + "=="), byteorder="big")
                        e = int.from_bytes(base64.urlsafe_b64decode(k["e"] + "=="), byteorder="big")
                        pub_numbers = rsa.RSAPublicNumbers(e, n)
                        key_to_use = pub_numbers.public_key(default_backend())
                        break
            if key_to_use is None:
                # Fallback to first PEM
                pem_source = (os.environ.get("JWT_PUBLIC_KEY") or "").split("||")[0]
                key_to_use = serialization.load_pem_public_key(pem_source.encode("utf-8"), backend=default_backend())
            jwt.decode(
                token,
                key_to_use,
                algorithms=["RS256"],
                options={"require": ["exp", "iat", "nbf"]},
                issuer=JWT_ISS,
                audience=JWT_AUD,
            )
        else:
            jwt.decode(
                token,
                JWT_SECRET,
                algorithms=["HS256"],
                options={"require": ["exp", "iat", "nbf"]},
                issuer=JWT_ISS,
                audience=JWT_AUD,
            )
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

# Live config reload
_config_cache: Dict[str, Any] = {}

class ConfigReloadHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith(CONFIG_PATH):
            with tracer.start_as_current_span("config_reload"):
                logger.info("Config file changed, reloading...")
                global _config_cache
                try:
                    import yaml
                    with open(CONFIG_PATH) as f:
                        new_config = yaml.safe_load(f) or {}
                    _config_cache.update(new_config)
                    logger.info("Config reloaded successfully: %s", {k: v for k, v in new_config.items() if not k.startswith('_')})
                except Exception as exc:
                    logger.error("Config reload failed: %s", exc)
# Track file observer for shutdown
FILE_OBSERVER = Observer()
FILE_OBSERVER.schedule(ConfigReloadHandler(), path=os.path.dirname(CONFIG_PATH) or '.', recursive=False)
FILE_OBSERVER.start()

class StatusResponse(BaseModel):
    model_version: str
    healthy: bool



# enhanced status endpoint
@app.get("/status", response_model=StatusResponse)
async def status():
    """Enhanced status with API & model version info and vocabulary summary."""
    metrics_summary = get_metrics_summary()

    return {
        "api_version": API_VERSION,
        "model_version": MODEL_VERSION,
        "healthy": True,
        "metrics": metrics_summary,
        "vocabulary": metrics_summary.get('vocabulary', {})
    }

TRUSTED_PROXIES = {h.strip() for h in os.environ.get("TRUSTED_PROXIES", "").split(",") if h.strip()}

def _get_client_id(request: Request) -> str:
    # Prefer explicit client ID header; fallback to remote IP
    hdr = request.headers.get('X-Client-ID')
    if hdr:
        return hdr
    # Use X-Forwarded-For only if request comes through a trusted proxy
    xff = request.headers.get('X-Forwarded-For')
    if xff and request.client and request.client.host in TRUSTED_PROXIES:
        return xff.split(',')[0].strip()
    return request.client.host if request.client else 'unknown'

@app.post("/decode")
async def decode(request: Request, x_target_language: str = Header(None), x_adapter_override: str = Header(None)):
    client_id = _get_client_id(request)
    allowed, message = rate_limiter.is_allowed(client_id)
    if not allowed:
        raise HTTPException(status_code=429, detail=message)
    start_time = time.time()
    active_connections.inc()  # Increment active connections

    try:
        with tracer.start_as_current_span("decode_request") as span:
            span.set_attribute("target_language", x_target_language or "unknown")
            
            # Get request data
            compressed_data = await request.body()
            
            # Decompress encoder output
            decompressed_data = decompress_encoder_output(compressed_data)
            
            # Get vocabulary pack for target language
            target_lang = x_target_language or "en"
            vocab_manager = get_vocabulary_manager()
            vocab_pack = vocab_manager.get_vocab_for_pair("en", target_lang)

            # Optional: override adapter (e.g., composed adapter name) for zero-shot
            current_model = get_model()
            adapter_to_use = x_adapter_override
            if adapter_to_use and isinstance(current_model, AdapterUniversalEncoder):
                try:
                    if adapter_to_use not in current_model.language_adapters:
                        # Composition should have registered it; skip I/O
                        pass
                    current_model.loaded_adapters.add(adapter_to_use)
                except Exception:
                    logger.debug("adapter_load_failed", exc_info=True)
                    adapter_to_use = None
            
            # Run decoder inference
            if current_model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            # Prepare input tensors
            dev = get_device()
            encoder_hidden_states = torch.tensor(decompressed_data['hidden_states'], device=dev)
            encoder_attention_mask = torch.tensor(decompressed_data['attention_mask'], device=dev)
            
            # Generate translation
            with torch.no_grad():
                output_tokens, _ = current_model.generate(
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    target_lang_id=vocab_pack.getTokenId(f"<{target_lang}>"),
                    max_length=128,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9
                )
            
            # Decode tokens to text
            translation = decode_tokens_to_text(output_tokens.cpu().numpy(), vocab_pack)
            
            result = {"translation": translation, "target_language": target_lang}

        # Track successful request
        latency = time.time() - start_time
        track_translation_request(
            source_lang="unknown",  # Will be extracted from encoder data
            target_lang=target_lang,
            status='success',
            latency=latency
        )
        
        return result
        
    except Exception as e:
        # Track failed request
        track_translation_request(
            source_lang="unknown",
            target_lang=x_target_language or "unknown",
            status='error'
        )
        logger.error(f"Decode error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        active_connections.dec()  # Decrement active connections

# JWKS endpoint (mirrored)
@app.get("/.well-known/jwks.json")
async def jwks_decoder():
    return JSONResponse({"keys": get_jwks_keys()})

# Health and readiness endpoints using routers
probe_router = APIRouter(tags=["Probes"])
metrics_router = APIRouter(tags=["Metrics"])
admin_router = APIRouter(tags=["Admin"]) 

@probe_router.get('/health')
async def decoder_health():
    """Liveness probe with build/version info."""
    from utils.service_health import load_version_info
    v = load_version_info()
    return {"status": "ok", "version": v.get("version"), "apiVersion": v.get("apiVersion")}

@probe_router.get('/ready')
async def decoder_ready():
    """Readiness probe: checks model, vocabulary manager, and JWKS (when RS256 configured)."""
    from utils.service_health import jwks_readiness, build_ready_payload

    version = load_version_info()
    jwks_keys = get_jwks_keys()
    jwks_ok = jwks_readiness(env=os.environ, jwks_keys=jwks_keys)

    checks = {
        "model": get_model() is not None,
        "vocabulary": get_vocabulary_manager() is not None,
        "jwks": jwks_ok,
    }
    payload = build_ready_payload(component="decoder", version=version, checks=checks)
    return JSONResponse(content=payload, status_code=200 if payload["ready"] else 503)

@metrics_router.get("/metrics")
async def decoder_metrics(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Prometheus metrics endpoint (protected)."""
    require_jwt(credentials)
    from prometheus_client import generate_latest
    collect_vocabulary_metrics()
    return Response(generate_latest(), media_type="text/plain")

# Admin endpoints
@admin_router.post("/admin/reload_model")
def reload_model(credentials: HTTPAuthorizationCredentials = Depends(require_jwt)):
    with tracer.start_as_current_span("reload_model"):
        # ... reload model logic ...
        return {"status": "Model reloaded"}

# Mount routers
app.include_router(probe_router)
app.include_router(metrics_router)
app.include_router(admin_router)

# OpenAPI and Swagger UI are available at /openapi.json and /docs
# All endpoints are traced, and admin endpoints require JWT

# Continuous batching for maximum GPU utilization

batcher = ContinuousBatcher()

# --- MODIFIED: Unified Global model and optimization ---
model: Optional[AdapterUniversalEncoder] = None # We will use the adapter-aware encoder
adapter_manager: Optional[AdapterManager] = None # Add a global manager
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocabulary_manager = VocabularyManager()
_MODEL_LOCK = threading.RLock()
JWKS_LOCK = threading.RLock()

def get_model():
    with _MODEL_LOCK:
        return model

def set_model(val):
    with _MODEL_LOCK:
        global model
        model = val

def get_adapter_manager():
    with _MODEL_LOCK:
        return adapter_manager

def get_device():
    with _MODEL_LOCK:
        return device

def get_vocabulary_manager():
    with _MODEL_LOCK:
        return vocabulary_manager

def set_jwks_keys(keys):
    with JWKS_LOCK:
        global JWKS_KEYS
        JWKS_KEYS = keys

def get_jwks_keys():
    with JWKS_LOCK:
        return JWKS_KEYS

# --- END MODIFICATION ---

@app.on_event("startup")
async def startup():
    # Fast path for tests/CI to avoid heavy startup
    if os.environ.get("UTS_TEST_MODE") == "1":
        logger.info("UTS_TEST_MODE=1 detected: skipping heavy decoder startup")
        return

    # Prefetch artifacts from Hugging Face (models/vocabs/adapters) if configured
    try:
        store = ArtifactStore()  # requires HF_HUB_REPO_ID (and HF_TOKEN if needed)

        # Ensure base decoder model exists locally
        store.ensure_model("production/decoder.pt")

        # Optional prefetch via env (comma-separated)
        packs = os.environ.get("PREFETCH_VOCAB_GROUPS", "").strip()
        if packs:
            for p in [x.strip() for x in packs.split(",") if x.strip()]:
                try:
                    store.ensure_vocab_pack(p)
                except Exception as e:
                    logger.warning(
                        "prefetch_vocab_pack_failed",
                        extra={"pack": p, "error": str(e)}
                    )
        adapters = os.environ.get("PREFETCH_ADAPTERS", "").strip()
        if adapters:
            for a in [x.strip() for x in adapters.split(",") if x.strip()]:
                try:
                    store.ensure_adapter(a)
                except Exception as e:
                    logger.warning(
                        "prefetch_adapter_failed",
                        extra={"adapter": a, "error": str(e)}
                    )
    except Exception as e:
        logger.warning(
            "artifact_prefetch_skipped_or_failed",
            extra={"error": str(e)}
        )
    
    # Load model
    decoder_model = OptimizedUniversalDecoder().to(get_device())
    # Load the base model (without any specific adapters loaded initially)
    # This assumes your production decoder is saved here
    base_model_path = str(RuntimeDirectoryManager().production_dir / "decoder.pt")
    decoder_model = AdapterUniversalEncoder(base_model_path=base_model_path).to(get_device())
    decoder_model.eval()

    # Initialize the adapter manager with the model instance
    adapter_manager = AdapterManager(model=decoder_model, repo_id=HF_HUB_REPO_ID, max_cache_size=5)

    # Compile with torch.compile for faster inference
    if torch.__version__ >= "2.0.0":
        decoder_model = torch.compile(decoder_model, mode="reduce-overhead")

    set_model(decoder_model)

    # Start batch processor
    asyncio.create_task(batcher.process_batches())

    logger.info(f"✅ Decoder with Dynamic Adapter Loading is ready on {get_device()}")
    if torch.cuda.is_available():
        logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")


# Endpoint for the coordinator
@app.get("/loaded_adapters", response_model=List[str])
async def get_loaded_adapters_endpoint():
    """Returns a list of adapters currently loaded in the GPU cache."""
    mgr = get_adapter_manager()
    if not mgr:
        raise HTTPException(status_code=503, detail="Adapter manager not initialized")
    return await mgr.get_loaded_adapters()

async def process_batch_gpu(batch: List[Dict]) -> List[Dict]:
    """Process a batch on GPU with dynamic adapter loading and correct tensor handling.

    Expects each item in `batch` to contain:
      - 'compressed_data': bytes (encoder output in custom compressed format)
      - 'target_lang': str
      - optional 'domain': str
    Returns a list of result dicts ordered to match the input `batch`.
    """
    current_model = get_model()
    if current_model is None:
        raise RuntimeError("Model is not initialized")
    current_adapter_manager = get_adapter_manager()
    if current_adapter_manager is None:
        raise RuntimeError("Adapter manager is not initialized")
    current_vocab_manager = get_vocabulary_manager()

    # Best-effort artifact ensure before heavy compute
    try:
        store = ArtifactStore()
    except Exception:
        logger.debug("artifact_store_creation_failed", exc_info=True)
        store = None

    # Group by adapter to minimize swaps
    requests_by_adapter: Dict[str, List[Dict[str, Any]]] = {}
    for i, item in enumerate(batch):
        tgt = item.get("target_lang", "en")
        dom = item.get("domain")
        adapter_name = f"{tgt}_{dom}" if dom else tgt
        requests_by_adapter.setdefault(adapter_name, []).append({
            "original_index": i,
            "data": item,
            "target_lang": tgt,
        })
        if store:
            try:
                store.ensure_for_language_pair("en", tgt, adapter=adapter_name)
            except Exception as e:
                logger.debug(f"ensure_for_language_pair skipped: {e}")

    all_results: List[Optional[Dict]] = [None] * len(batch)

    # Process each adapter group
    for adapter_name, requests in requests_by_adapter.items():
        # Ensure adapter is loaded (async)
        await current_adapter_manager.get_adapter(adapter_name)

        # Decompress and collect tensors
        hidden_list: List[torch.Tensor] = []
        mask_list: List[torch.Tensor] = []
        tgt_langs: List[str] = []
        orig_indices: List[int] = []
        for r in requests:
            decomp = decompress_encoder_output(r["data"]["compressed_data"])  # tensors on correct device
            hidden_list.append(decomp["hidden_states"])   # [1, T, H]
            mask_list.append(decomp["attention_mask"])    # [1, T]
            tgt_langs.append(r["target_lang"])            # e.g., 'es'
            orig_indices.append(r["original_index"])      # original position in batch

        # Concatenate along batch dimension
        encoder_hidden = torch.cat(hidden_list, dim=0)            # [B, T, H]
        encoder_mask = torch.cat(mask_list, dim=0)                 # [B, T]

        # Choose target language id (group should be homogeneous)
        target_lang_ids = [current_vocab_manager.language_to_pack.get(lang, 3) for lang in tgt_langs]
        target_lang_id = target_lang_ids[0]

        # Generate outputs
        with torch.no_grad():
            generated_ids, _scores = current_model.generate(
                encoder_hidden,
                encoder_mask,
                target_lang_id,
                max_length=128,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
            )

        # Decode per-sample and place back in original order
        for i in range(generated_ids.size(0)):
            vocab_pack = current_vocab_manager.get_vocab_for_pair("en", tgt_langs[i])
            tokens_np = generated_ids[i].detach().cpu().numpy()
            text = decode_tokens_to_text(tokens_np, vocab_pack)
            all_results[orig_indices[i]] = {
                "translation": text,
                "target_lang": tgt_langs[i],
            }

    return all_results


# (decompress_encoder_output and decode_tokens_to_text moved to decoder_core.py)

#Endpoint for adapter composition 
@app.post("/compose_adapter", status_code=201, dependencies=[Depends(verify_internal_request)])
async def compose_adapter_endpoint(request: CompositionRequest):
    """
    Triggers the on-the-fly creation of a composed adapter for zero-shot pairs.
    This endpoint is protected and only accessible by internal services like the coordinator.
    """
    current_model = get_model()
    current_adapter_manager = get_adapter_manager()
    if not current_model or not isinstance(current_model, AdapterUniversalEncoder):
        raise HTTPException(status_code=503, detail="Model not initialized or does not support adapters.")
    
    try:
        # Ensure the base adapters are loaded first (await async operations)
        await current_adapter_manager.get_adapter(request.source_adapter)
        await current_adapter_manager.get_adapter(request.target_adapter)
        
        # Perform the composition
        composed_name = current_model.compose_adapters(
            source_adapter_name=request.source_adapter,
            target_adapter_name=request.target_adapter,
            composition_strategy=request.strategy
        )
        return {"status": "success", "composed_adapter_name": composed_name}
    except Exception as e:
        logger.error(f"Adapter composition failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to compose adapters: {str(e)}")

# Deployment configuration for Kubernetes
"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: universal-decoder
spec:
  replicas: 2
  selector:
    matchLabels:
      app: universal-decoder
  template:
    metadata:
      labels:
        app: universal-decoder
    spec:
      containers:
      - name: decoder
        image: your-registry/universal-decoder:latest
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
          limits:
            memory: "16Gi"
            cpu: "8"
            nvidia.com/gpu: 1
        env:
        - name: OMP_NUM_THREADS
          value: "4"
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
      nodeSelector:
        gpu-type: "t4"
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT, workers=API_WORKERS)