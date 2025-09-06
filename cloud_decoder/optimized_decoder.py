# cloud_decoder/optimized_decoder.py
import torch
import torch.nn as nn
from pathlib import Path
import struct
import numpy as np
import litserve as ls
from fastapi import FastAPI, Request, Header, Depends, HTTPException
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
import triton_python_backend_utils as pb_utils
import msgpack
import lz4.frame
import logging
import os
import yaml
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from monitoring.metrics_collector import (
    track_translation_request,
    collect_vocabulary_metrics,
    get_metrics_summary,
    active_connections,
    gpu_utilization
)
from utils.logging_config import setup_logging
from collections import OrderedDict
# Import vocabulary manager
from vocabulary.unified_vocab_manager import UnifiedVocabularyManager, VocabularyMode

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
JWT_SECRET = os.environ.get("DECODER_JWT_SECRET")
if not JWT_SECRET or JWT_SECRET in {"jwtsecret123", "use-openssl-rand-hex-32-to-generate-a-secure-key"}:
    raise RuntimeError("DECODER_JWT_SECRET must be set to a strong, random value.")
CONFIG_PATH = os.environ.get("DECODER_CONFIG_PATH", "config/decoder_config.yaml")
HF_HUB_REPO_ID = os.environ.get("HF_HUB_REPO_ID", "your-hf-org/universal-translation-system")

# API endpoints and server configuration
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))
API_WORKERS = int(os.environ.get("API_WORKERS", "1"))
API_TITLE = os.environ.get("API_TITLE", "Cloud Decoder API")

# Initialize logging (centralized)
setup_logging(log_dir="logs", log_level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("decoder")

# Initialize utilities
api_key_manager = APIKeyManager()
rate_limiter = RateLimiter(requests_per_minute=60, requests_per_hour=1000)

# AdapterManager Class 
class AdapterManager:
    """
    Manages dynamic loading and LRU caching of language adapters on a decoder node.
    """
    def __init__(self, model: nn.Module, repo_id: str, max_cache_size: int = 5, adapter_dir: str = "models/adapters"):
        self.model = model  # The AdapterUniversalEncoder instance
        self.adapter_dir = Path(adapter_dir)
        self.max_cache_size = max_cache_size
        self.repo_id = repo_id
        
        # Use an OrderedDict for a simple and effective LRU cache
        # Maps adapter_name -> loaded adapter module
        self.cache = OrderedDict()
        # --- MODIFIED: Use asyncio.Lock for async environments ---
        self.lock = asyncio.Lock()
        self.loading_events: Dict[str, asyncio.Event] = {}
        
        logger.info(f"AdapterManager initialized for repo '{self.repo_id}' with cache size: {self.max_cache_size}")

    async def get_adapter(self, adapter_name: str):
        """
        Asynchronously retrieves a language adapter, loading it from disk if not in the cache.
        This is now non-blocking for high-concurrency environments.
        """
        # --- Non-blocking cache check ---
        async with self.lock:
            if adapter_name in self.cache: # Cache Hit
                self.cache.move_to_end(adapter_name) # Mark as recently used
                logger.debug(f"Adapter cache hit for '{adapter_name}'")
                return

            # --- Cache Miss ---
            # Check if another thread is already loading this adapter
            if adapter_name in self.loading_events:
                event = self.loading_events[adapter_name]
                # Release the main lock and wait for the other thread to finish
                async with self.lock.released():
                    logger.info(f"Adapter '{adapter_name}' is being loaded by another task, waiting...")
                    await event.wait()
                # After waiting, the adapter should be in the cache. Re-acquire lock and check again.
                async with self.lock:
                    if adapter_name in self.cache:
                        self.cache.move_to_end(adapter_name)
                        return
                    else: # Should not happen, but as a fallback
                        logger.error(f"Waited for adapter '{adapter_name}' but it was not loaded.")
                        raise RuntimeError(f"Failed to load adapter {adapter_name}")

            # This is the first thread to request this adapter, so it will load it.
            # Create an event to signal other threads.
            self.loading_events[adapter_name] = asyncio.Event()

        # --- Perform slow I/O outside the main lock ---
        try:
            # Download the adapter from the Hub if it doesn't exist locally
            await self._download_adapter_from_hub(adapter_name)

            # --- Re-acquire lock to update the shared cache state ---
            async with self.lock:
                logger.info(f"Adapter cache miss for '{adapter_name}'. Loading from disk...")
                # Evict if cache is full
                if len(self.cache) >= self.max_cache_size:
                    oldest_adapter_name, _ = self.cache.popitem(last=False)
                    if oldest_adapter_name in self.model.language_adapters:
                        del self.model.language_adapters[oldest_adapter_name]
                    logger.info(f"Evicted adapter '{oldest_adapter_name}' from cache.")

                # Load the new adapter into the model
                adapter_path = self.adapter_dir / f"best_{adapter_name}_adapter.pt"
                self.model.load_language_adapter(adapter_name, str(adapter_path))
                self.cache[adapter_name] = self.model.language_adapters[adapter_name]
                logger.info(f"Successfully loaded and cached adapter '{adapter_name}'.")
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
            logger.debug(f"Adapter '{adapter_name}' already exists locally at {local_adapter_path}")
            return

        logger.info(f"Adapter '{adapter_name}' not found locally. Downloading from Hugging Face Hub repo: {self.repo_id}")
        
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
            logger.info(f"✅ Successfully downloaded adapter '{adapter_name}' to {local_adapter_path}")
        except HfHubHTTPError as e:
            if e.response.status_code == 404:
                logger.error(f"❌ Adapter '{repo_filename}' not found in repo '{self.repo_id}'.")
                raise FileNotFoundError(f"Adapter not found on Hub: {repo_filename}") from e
            logger.error(f"❌ HTTP error downloading adapter '{adapter_name}': {e}")
            raise IOError(f"Failed to download adapter: {e}") from e
        except Exception as e:
            logger.error(f"❌ An unexpected error occurred during adapter download: {e}")
            raise IOError(f"Failed to download adapter: {e}") from e

    async def get_loaded_adapters(self) -> List[str]:
        """Returns a list of adapters currently hot in the cache."""
        async with self.lock:
            return list(self.cache.keys())
            # Evict the least recently used adapter if the cache is full
            if len(self.cache) >= self.max_cache_size:
                oldest_adapter_name, _ = self.cache.popitem(last=False)
                # Remove from the model's ModuleDict to free memory
                if oldest_adapter_name in self.model.language_adapters:
                    del self.model.language_adapters[oldest_adapter_name]
                logger.info(f"Evicted adapter '{oldest_adapter_name}' from cache.")

            # Load the new adapter
            adapter_path = self.adapter_dir / f"best_{adapter_name}_adapter.pt"
            if not adapter_path.exists():
                logger.error(f"Adapter file not found: {adapter_path}")
                # Don't add to cache if not found
                return

            # Use the model's own method to add and load the adapter
            self.model.load_language_adapter(adapter_name, str(adapter_path))
            
            # Add the newly loaded adapter module to our cache
            self.cache[adapter_name] = self.model.language_adapters[adapter_name]
            logger.info(f"Successfully loaded and cached adapter '{adapter_name}'.")

    def get_loaded_adapters(self) -> List[str]:
        """Returns a list of adapters currently hot in the cache."""
        with self.lock:
            return list(self.cache.keys())

class OptimizedUniversalDecoder(nn.Module):
    """
    Custom decoder optimized for low-end GPUs (T4, RTX 3060)
    Designed to maximize throughput with minimal memory
    """
    
    def __init__(
        self,
        encoder_dim: int = 1024,
        decoder_dim: int = 512,  # Smaller than encoder for efficiency
        num_layers: int = 6,
        num_heads: int = 8,
        vocab_size: int = 50000,
        max_length: int = 256,
        device: torch.device = None
    ):
        super().__init__()
        
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.device = device or torch.device('cpu')
        
        # Dynamic embeddings (loaded per request batch)
        self.embedding = nn.Embedding(vocab_size, decoder_dim)
        self.positional_embedding = nn.Embedding(max_length, decoder_dim)
        
        # Efficient encoder adapter
        self.encoder_adapter = nn.Linear(encoder_dim, decoder_dim, bias=False)
        
        # Optimized decoder layers
        self.layers = nn.ModuleList([
            OptimizedDecoderLayer(
                decoder_dim=decoder_dim,
                num_heads=num_heads,
                dropout=0.1
            ) for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(decoder_dim)
        
        # Tied embeddings for output
        self.output_projection = nn.Linear(decoder_dim, vocab_size, bias=False)
        self.output_projection.weight = self.embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
 
    @torch.jit.script_method
    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        compressed_embeddings: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """Optimized forward pass with torch.jit support"""
        
        batch_size, seq_len = decoder_input_ids.shape
        device = decoder_input_ids.device

        # Handle compressed embeddings if provided
        if compressed_embeddings is not None:
            encoder_hidden_states = self.decompress_embeddings(compressed_embeddings)
        
        # Embeddings
        x = self.embedding(decoder_input_ids)
        positions = torch.arange(seq_len, device=device).expand(batch_size, -1)
        x = x + self.positional_embedding(positions)
        
        # Adapt encoder hidden states
        encoder_hidden = self.encoder_adapter(encoder_hidden_states)
        
        # Causal mask for decoder
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1
        )
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, encoder_hidden, causal_mask, encoder_attention_mask)
        
        x = self.layer_norm(x)
        
        # Output projection
        logits = self.output_projection(x)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        target_lang_id: int,
        max_length: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int = 2,
        pad_token_id: int = 0
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Optimized generation with multiple decoding strategies
        Returns: (generated_ids, scores)
        """
        batch_size = encoder_hidden_states.size(0)
        device = encoder_hidden_states.device
        
        # Initialize with target language token
        decoder_input_ids = torch.full((batch_size, 1), target_lang_id, device=device)
        
        # Track finished sequences
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        scores = []
        
        for step in range(max_length - 1):
            # Forward pass
            logits = self.forward(
                decoder_input_ids,
                encoder_hidden_states,
                encoder_attention_mask
            )
            
            # Get next token logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)

            # Update finished sequences
            finished = finished | (next_tokens.squeeze(-1) == eos_token_id)
            
            # Replace tokens for finished sequences with padding
            next_tokens[finished] = pad_token_id
            
            # Append to sequence
            decoder_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=1)

            # Track scores
            token_scores = torch.gather(probs, 1, next_tokens).squeeze(-1)
            scores.append(token_scores.cpu().tolist())
            
            # Check if all sequences are finished
            if finished.all():
                break
        
        return decoder_input_ids, scores


class OptimizedDecoderLayer(nn.Module):
    """Single decoder layer optimized for GPU efficiency"""
    
    def __init__(self, decoder_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            decoder_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(decoder_dim)
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            decoder_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(decoder_dim)
        
        # Optimized FFN with SwiGLU activation
        self.ffn = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim * 4),
            nn.SiLU(),
            nn.Linear(decoder_dim * 4, decoder_dim)
        )
        self.ffn_norm = nn.LayerNorm(decoder_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden: torch.Tensor,
        causal_mask: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention
        residual = x
        x = self.self_attn_norm(x)
        x, _ = self.self_attn(x, x, x, attn_mask=causal_mask, is_causal=True)
        x = residual + self.dropout(x)
        
        # Cross-attention
        residual = x
        x = self.cross_attn_norm(x)
        x, _ = self.cross_attn(x, encoder_hidden, encoder_hidden, key_padding_mask=encoder_mask)
        x = residual + self.dropout(x)
        
        # FFN
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)
        
        return x


# Pydantic model for the request body
class CompositionRequest(BaseModel):
    source_adapter: str
    target_adapter: str
    strategy: str = "average"

# FastAPI application for serving
app = FastAPI(title=API_TITLE, version=MODEL_VERSION, openapi_url="/openapi.json")
FastAPIInstrumentor.instrument_app(app)

@app.on_event("startup")
async def startup_validation():
    # Validate critical env on startup
    missing = []
    def _check(name: str, val: Optional[str]):
        if not val: missing.append(name)
    # Support Docker secrets: if DECODER_JWT_SECRET_FILE is set, read from file
    secret_file = os.environ.get("DECODER_JWT_SECRET_FILE")
    if secret_file and os.path.exists(secret_file):
        try:
            with open(secret_file, 'r', encoding='utf-8') as f:
                os.environ['DECODER_JWT_SECRET'] = f.read().strip()
        except Exception as ex:
            logger.error(f"Failed to read DECODER_JWT_SECRET_FILE: {ex}")
    _check("DECODER_JWT_SECRET", os.environ.get("DECODER_JWT_SECRET"))
    # RS256 for decoder admin endpoints is optional; if private key is set, require public key too
    if os.environ.get("JWT_PRIVATE_KEY") and not os.environ.get("JWT_PUBLIC_KEY"):
        raise RuntimeError("JWT_PRIVATE_KEY set but JWT_PUBLIC_KEY missing for decoder")
    if missing:
        logger.error(f"Missing required env: {', '.join(missing)}")
        raise RuntimeError("Decoder startup validation failed: missing secrets")

    # Build JWKS using shared utility
    from utils.jwks_utils import build_jwks_from_env, diff_kids
    global JWKS_KEYS
    JWKS_KEYS = build_jwks_from_env(component="decoder", env=os.environ)

    # Background job to periodically reload JWKS every 5 minutes, with change logging
    async def jwks_reload_loop_decoder():
        global JWKS_KEYS
        last_keys = JWKS_KEYS
        while True:
            try:
                new_keys = build_jwks_from_env(component="decoder", env=os.environ)
                if new_keys:
                    added, removed = diff_kids(last_keys, new_keys)
                    JWKS_KEYS = new_keys
                    if added or removed:
                        logger.info(f"Decoder JWKS changed. Added: {added or '[]'}, Removed: {removed or '[]'}")
                    last_keys = new_keys
            except Exception as ex:
                logger.debug(f"Decoder JWKS reload loop error: {ex}")
            await asyncio.sleep(300)

    asyncio.create_task(jwks_reload_loop_decoder())

# CORS and Security headers middleware
from fastapi.middleware.cors import CORSMiddleware

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
            if kid and JWKS_KEYS:
                for k in JWKS_KEYS:
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
class ConfigReloadHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith(CONFIG_PATH):
            with tracer.start_as_current_span("config_reload"):
                print("[Decoder] Config file changed, reloading...")
                # Reload config logic here
                # (For demo, just print. In production, update in-memory config.)
observer = Observer()
observer.schedule(ConfigReloadHandler(), path=os.path.dirname(CONFIG_PATH) or '.', recursive=False)
observer.start()

class StatusResponse(BaseModel):
    model_version: str
    healthy: bool

# basic health endpoint (no sensitive info)
@app.get("/health")
async def health():
    return {"status": "ok"}

# enhanced status endpoint
@app.get("/status", response_model=StatusResponse)
async def status():
    """Enhanced status with vocabulary info."""
    metrics_summary = get_metrics_summary()

    return {
        "model_version": MODEL_VERSION,
        "healthy": True,
        "metrics": metrics_summary,
        "vocabulary": metrics_summary.get('vocabulary', {})
    }

    with tracer.start_as_current_span("status_endpoint") as span:
        # Get vocabulary version info
        vocab_versions = vocabulary_manager.get_vocabulary_version_info()

        span.set_attribute("model_version", MODEL_VERSION)
        span.set_attribute("vocabulary_packs", json.dumps(vocab_versions))

        return {"model_version": MODEL_VERSION, "healthy": True, "vocabulary_packs": vocab_versions}

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
async def decode(request: Request, x_target_language: str = Header(None)):
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
            vocab_pack = vocabulary_manager.get_vocab_for_pair("en", target_lang)
            
            # Run decoder inference
            if model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            # Prepare input tensors
            encoder_hidden_states = torch.tensor(decompressed_data['hidden_states'], device=device)
            encoder_attention_mask = torch.tensor(decompressed_data['attention_mask'], device=device)
            
            # Generate translation
            with torch.no_grad():
                output_tokens, _ = model.generate(
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
    return JSONResponse({"keys": JWKS_KEYS})

# metrics endpoint (protected)
@app.get("/metrics")
async def metrics(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Prometheus metrics endpoint (protected)."""
    # Validate JWT
    require_jwt(credentials)
    from prometheus_client import generate_latest
    
    # Collect vocabulary metrics before generating response
    collect_vocabulary_metrics()
    
    return Response(generate_latest(), media_type="text/plain")        

@app.post("/admin/reload_model")
def reload_model(credentials: HTTPAuthorizationCredentials = Depends(require_jwt)):
    with tracer.start_as_current_span("reload_model"):
        # ... reload model logic ...
        return {"status": "Model reloaded"}

# OpenAPI and Swagger UI are available at /openapi.json and /docs
# All endpoints are traced, and admin endpoints require JWT

# Continuous batching for maximum GPU utilization
class ContinuousBatcher:
    def __init__(self, max_batch_size: int = 64, timeout_ms: int = 10):
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.pending_requests = asyncio.Queue()
        self.processing = False
        
    async def add_request(self, request_data: Dict) -> Dict:
        future = asyncio.Future()
        await self.pending_requests.put((request_data, future))
        return await future
    
    async def process_batches(self):
        """Continuous batch processing for maximum GPU utilization"""
        while True:
            batch = []
            futures = []
            
            # Collect requests up to batch size or timeout
            start_time = time.time()
            while len(batch) < self.max_batch_size:
                try:
                    timeout = self.timeout_ms / 1000 - (time.time() - start_time)
                    if timeout <= 0:
                        break
                        
                    request_data, future = await asyncio.wait_for(
                        self.pending_requests.get(),
                        timeout=timeout
                    )
                    batch.append(request_data)
                    futures.append(future)
                except asyncio.TimeoutError:
                    break
            
            if batch:
                # Process batch on GPU
                try:
                    results = await process_batch_gpu(batch)
                    for future, result in zip(futures, results):
                        future.set_result(result)
                except Exception as e:
                    for future in futures:
                        future.set_exception(e)

batcher = ContinuousBatcher()

# --- MODIFIED: Unified Global model and optimization ---
model: Optional[AdapterUniversalEncoder] = None # We will use the adapter-aware encoder
adapter_manager: Optional[AdapterManager] = None # Add a global manager
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocabulary_manager = VocabularyManager()
# --- END MODIFICATION ---

@app.on_event("startup")
async def startup():
    global model, adapter_manager

    # Prefetch artifacts from Hugging Face (models/vocabs/adapters) if configured
    try:
        from utils.artifact_store import ArtifactStore
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
                    logger.warning(f"Prefetch vocab pack '{p}' failed: {e}")
        adapters = os.environ.get("PREFETCH_ADAPTERS", "").strip()
        if adapters:
            for a in [x.strip() for x in adapters.split(",") if x.strip()]:
                try:
                    store.ensure_adapter(a)
                except Exception as e:
                    logger.warning(f"Prefetch adapter '{a}' failed: {e}")
    except Exception as e:
        logger.warning(f"Artifact prefetch skipped or failed: {e}")
    
    # Load model
    model = OptimizedUniversalDecoder().to(device)
    # Load the base model (without any specific adapters loaded initially)
    # This assumes your production decoder is saved here
    base_model_path = "models/production/decoder.pt"
    model = AdapterUniversalEncoder(base_model_path=base_model_path).to(device)
    model.eval()

    # Initialize the adapter manager with the model instance
    adapter_manager = AdapterManager(model=model, repo_id=HF_HUB_REPO_ID, max_cache_size=5)

    # Compile with torch.compile for faster inference
    if torch.__version__ >= "2.0.0":
        model = torch.compile(model, mode="reduce-overhead")

    # Start batch processor
    asyncio.create_task(batcher.process_batches())

    logger.info(f"✅ Decoder with Dynamic Adapter Loading is ready on {device}")
    if torch.cuda.is_available():
        logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")


@app.post("/decode")
async def decode(request: Request):
    """
    Decode encoder output to target language
    """
    try:
        # Read compressed data
        compressed_data = await request.body()

        # Add to batch
        result = await batcher.add_request({
            'compressed_data': compressed_data,
            'target_lang': request.headers.get('x-target-language'),
            'domain': request.headers.get('x-domain')
        })

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Decode error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.route('/health', methods=['GET'])
async def health(request):
    """Health check endpoint"""
    return {"status": "healthy", "device": str(device)}

# +++ ADDED: New endpoint for the coordinator +++
@app.get("/loaded_adapters", response_model=List[str])
async def get_loaded_adapters_endpoint():
    """Returns a list of adapters currently loaded in the GPU cache."""
    if not adapter_manager:
        raise HTTPException(status_code=503, detail="Adapter manager not initialized")
    return adapter_manager.get_loaded_adapters()    

async def process_batch_gpu(batch: List[Dict]) -> List[Dict]:
    """Process batch on GPU with maximum efficiency and dynamic adapter loading"""

    # Ensure artifacts exist for each target language/domain before heavy work
    try:
        from utils.artifact_store import ArtifactStore
        store = ArtifactStore()
    except Exception:
        store = None

    # Group requests by target language to minimize adapter swapping
    requests_by_lang = defaultdict(list)
    for i, item in enumerate(batch):
        # Determine the adapter needed (e.g., 'es' or 'es_medical')
        target_lang = item.get('target_lang', 'en')
        domain = item.get('domain')
        adapter_name = f"{target_lang}_{domain}" if domain else target_lang
        requests_by_lang[adapter_name].append({'original_index': i, 'data': item})
        # Best-effort ensure packs/adapters
        if store:
            try:
                store.ensure_for_language_pair('en', target_lang, adapter=adapter_name)
            except Exception as e:
                logger.debug(f"ensure_for_language_pair skipped: {e}")

    all_results = [None] * len(batch)

    for adapter_name, requests in requests_by_lang.items():
        # --- DYNAMIC LOADING HAPPENS HERE ---
        adapter_manager.get_adapter(adapter_name)

        # Prepare the sub-batch for this language/adapter
        # ... (your existing batch preparation logic: decompress, stack tensors) ...

        with torch.no_grad(), torch.cuda.amp.autocast():
            # Pass the adapter name to the forward pass
            encoder_output = model(encoder_hidden, attention_mask, language=adapter_name)
            # ... (your existing generation and decoding logic) ...

        # Place results back in the correct order
        for i, result in enumerate(sub_batch_results):
            original_index = requests[i]['original_index']
            all_results[original_index] = result
            
    return all_results    
    
    with torch.cuda.amp.autocast():  # Mixed precision for speed
        # Prepare batch tensors
        encoder_outputs = []
        encoder_masks = []
        target_langs = []
        
        for item in batch:
            # Decompress encoder output
            decompressed = decompress_encoder_output(item['compressed_data'])

            encoder_outputs.append(decompressed['hidden_states'])
            encoder_masks.append(decompressed['attention_mask'])
            target_langs.append(item['target_lang'])
        
        # Stack tensors
        encoder_hidden = torch.tensor(np.stack(encoder_outputs)).to(device)
        encoder_mask = torch.tensor(np.stack(encoder_masks)).to(device)
        
        # Get target language IDs
        target_lang_ids = [vocabulary_manager.language_to_pack.get(lang, 3) for lang in target_langs]
        
        # Generate translations
        with torch.no_grad():
            output_ids = model.generate(
                encoder_hidden,
                encoder_mask,
                target_lang_ids[0],  # Assuming same target lang for batch
                max_length=128,
                temperature=0.7,
                top_k=50,
                top_p=0.9
            )
        
        # Decode to text
        results = []
        for i, output in enumerate(output_ids):
            # Get vocabulary pack for target language
            vocab_pack = vocabulary_manager.get_vocab_for_pair('en', target_langs[i])

            # Decode tokens to text
            tokens = output.cpu().numpy()
            text = decode_tokens_to_text(tokens, vocab_pack)

            results.append({
                'translation': text,
                'target_lang': target_langs[i],
                'scores': scores if scores else None
            })
        
        return results


def decompress_encoder_output(self, compressed_data: bytes) -> Dict:
    """Decompress encoder output with correct 16-byte header parsing.
    Expects bytes; raises if provided a different type.
    """
    if not isinstance(compressed_data, (bytes, bytearray)):
        raise ValueError("decompress_encoder_output expects bytes/bytearray")

    header_size = 16  # 4*int32 + 4*float32
    if len(compressed_data) < header_size:
        raise ValueError("Invalid compressed data: header is too short.")

    # Unpack the 16-byte header using struct: seq_len (i), hidden_dim (i), scale (f), reserved (i)
    seq_len, hidden_dim, scale, _ = struct.unpack('<iifi', compressed_data[:header_size])

    # Decompress the payload
    compressed_payload = compressed_data[header_size:]
    decompressed_payload = lz4.frame.decompress(compressed_payload)

    # Dequantize the int8 data to float32
    quantized_embeddings = np.frombuffer(decompressed_payload, dtype=np.int8)
    dequantized_embeddings = quantized_embeddings.astype(np.float32) * (1.0 / scale)

    # Reshape to the original 3D tensor shape
    hidden_states = dequantized_embeddings.reshape(1, seq_len, hidden_dim)

    return {
        'hidden_states': torch.tensor(hidden_states, device=self.device),
        'attention_mask': torch.ones((1, seq_len), dtype=torch.long, device=self.device)
    }

def decode_tokens_to_text(tokens: np.ndarray, vocab_pack) -> str:
    """Decode token IDs to text using vocabulary pack (production-grade)."""
    # Use SentencePiece if available
    try:
        import sentencepiece as spm
        if hasattr(vocab_pack, 'sp_model'):
            return vocab_pack.sp_model.decode_ids(list(tokens))
    except ImportError:
        pass
    # Fallback: reconstruct text using vocab pack
    id_to_token = {idx: tok for tok, idx in vocab_pack.tokens.items()}
    text_tokens = []
    for token_id in tokens:
        if token_id == 2:  # EOS token
            break
        elif token_id == 0:  # Padding
            continue
        token = id_to_token.get(token_id, '<unk>')
        text_tokens.append(token)
    # Join tokens and clean up subwords
    text = ' '.join(text_tokens)
    text = text.replace(' ##', '')
    text = text.replace('▁', ' ')
    return text.strip()

#Endpoint for adapter composition 
@app.post("/compose_adapter", status_code=201, dependencies=[Depends(verify_internal_request)])
async def compose_adapter_endpoint(request: CompositionRequest):
    """
    Triggers the on-the-fly creation of a composed adapter for zero-shot pairs.
    This endpoint is protected and only accessible by internal services like the coordinator.
    """
    if not model or not isinstance(model, AdapterUniversalEncoder):
        raise HTTPException(status_code=503, detail="Model not initialized or does not support adapters.")
    
    try:
        # Ensure the base adapters are loaded first
        adapter_manager.get_adapter(request.source_adapter)
        adapter_manager.get_adapter(request.target_adapter)
        
        # Perform the composition
        composed_name = model.compose_adapters(
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