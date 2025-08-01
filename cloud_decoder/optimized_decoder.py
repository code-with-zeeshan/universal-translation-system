# cloud_decoder/optimized_decoder.py
import torch
import torch.nn as nn
import struct
import numpy as np
import litserve as ls
from fastapi import FastAPI, Request, Header, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import asyncio
from typing import List, Dict, Optional, Tuple
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

# Import vocabulary manager
from vocabulary.vocabulary_manager import VocabularyManager

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

MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.0.0")
JWT_SECRET = os.environ.get("DECODER_JWT_SECRET", "jwtsecret123")
CONFIG_PATH = os.environ.get("DECODER_CONFIG_PATH", "config/decoder_config.yaml")

logger = logging.getLogger(__name__)

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


# FastAPI application for serving
app = FastAPI(title="Cloud Decoder API", version=MODEL_VERSION, openapi_url="/openapi.json")
FastAPIInstrumentor.instrument_app(app)

security = HTTPBearer()
def require_jwt(credentials: HTTPAuthorizationCredentials = Depends(security)):
    import jwt
    token = credentials.credentials
    try:
        jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
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

@app.post("/decode")
async def decode(request: Request, x_target_language: str = Header(None)):
    start_time = time.time()
    active_connections.inc()  # Increment active connections

    try:
        with tracer.start_as_current_span("decode_request") as span:
            span.set_attribute("target_language", x_target_language or "unknown")
            # ... actual decode logic ...
            return {"translation": "TODO: implement decode logic"}

        # Track successful request
        latency = time.time() - start_time
        track_translation_request(
            source_lang=source_lang,
            target_lang=target_lang,
            status='success',
            latency=latency
        )
        
        return result
        
    except Exception as e:
        # Track failed request
        track_translation_request(
            source_lang=source_lang,
            target_lang=target_lang,
            status='error'
        )
        raise
    finally:
        active_connections.dec()  # Decrement active connections

# metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
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

# Global model and optimization
model: Optional[OptimizedUniversalDecoder] = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocabulary_manager = VocabularyManager()

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


@app.on_event("startup")
async def startup():
    global model
    
    # Load model
    model = OptimizedUniversalDecoder().to(device)
    model.eval()
    
    # Compile with torch.compile for faster inference
    if torch.__version__ >= "2.0.0":
        model = torch.compile(model, mode="reduce-overhead")
    
    # Start batch processor
    asyncio.create_task(batcher.process_batches())
    
    logger.info(f"✅ Decoder ready on {device}")
    if torch.cuda.is_available():
        logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")


@app.route('/decode', methods=['POST'])
async def decode(request):
    """
    Decode encoder output to target language
    """
    try:
        # Read compressed data
        compressed_data = await request.body()
        
        # Add to batch
        result = await batcher.add_request({
            'compressed_data': compressed_data,
            'target_lang': request.headers.get('x-target-language')
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Decode error: {e}")
        return {"error": str(e)}

@app.route('/health', methods=['GET'])
async def health(request):
    """Health check endpoint"""
    return {"status": "healthy", "device": str(device)}

async def process_batch_gpu(batch: List[Dict]) -> List[Dict]:
    """Process batch on GPU with maximum efficiency"""
    
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
    """Decompress encoder output with correct 16-byte header parsing."""
    if not isinstance(compressed_data, bytes):
        # Handle non-byte data (e.g., already decompressed dict)
        return compressed_data

    header_size = 16  # 4*int32 + 4*float32
    if len(compressed_data) < header_size:
        raise ValueError("Invalid compressed data: header is too short.")

    # Unpack the 16-byte header using struct
    seq_len, hidden_dim, scale, _ = struct.unpack('<iif i', compressed_data[:header_size])    
        
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
    else:
        # Handle dict format
        return compressed_data

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
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)