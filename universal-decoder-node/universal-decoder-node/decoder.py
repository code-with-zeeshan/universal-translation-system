# universal-decoder-node/universal_decoder_node/decoder.py
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, Request, Header, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import asyncio
from typing import List, Dict, Optional, Tuple, Any
import time
import msgpack
import lz4.frame
import logging
import os
import yaml
import threading
from pathlib import Path
import jwt
import uvicorn

# Import vocabulary manager (we'll create a minimal version)
from .vocabulary import VocabularyManager

# Import utility modules
from .utils.auth import APIKeyManager
from .utils.rate_limiter import RateLimiter
from .utils.security import validate_model_source, safe_load_model

logger = logging.getLogger(__name__)

# Initialize utilities
api_key_manager = APIKeyManager()
rate_limiter = RateLimiter(requests_per_minute=60, requests_per_hour=1000)


class OptimizedUniversalDecoder(nn.Module):
    """
    Custom decoder optimized for low-end GPUs (T4, RTX 3060)
    Designed to maximize throughput with minimal memory
    """
    
    def __init__(
        self,
        encoder_dim: int = 1024,
        decoder_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        vocab_size: int = 50000,
        max_length: int = 256,
        device: torch.device = None
    ):
        super().__init__()
        
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Dynamic embeddings
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

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        compressed_embeddings: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """Optimized forward pass"""
        
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
        """Optimized generation with multiple decoding strategies"""
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

    def decompress_embeddings(self, compressed_data: Dict[str, Any]) -> torch.Tensor:
        """Decompress encoder embeddings"""
        if isinstance(compressed_data, dict):
            compressed_embeddings = compressed_data.get('embeddings')
            scale = compressed_data.get('scale', 1.0)
            original_shape = compressed_data.get('shape')
        else:
            return self._decompress_raw_embeddings(compressed_data)
        
        if isinstance(compressed_embeddings, bytes):
            decompressed = lz4.frame.decompress(compressed_embeddings)
            embeddings = np.frombuffer(decompressed, dtype=np.float32)
        else:
            embeddings = compressed_embeddings
        
        if original_shape:
            embeddings = embeddings.reshape(original_shape)
        
        return torch.tensor(embeddings, device=self.device) * scale

    def _decompress_raw_embeddings(self, compressed_data: bytes) -> torch.Tensor:
        """Decompress raw embedding data"""
        metadata_size = 12
        shape1 = int.from_bytes(compressed_data[0:4], 'little')
        shape2 = int.from_bytes(compressed_data[4:8], 'little')
        scale = np.frombuffer(compressed_data[8:12], dtype=np.float32)[0]
        
        compressed = compressed_data[metadata_size:]
        decompressed = lz4.frame.decompress(compressed)
        
        quantized = np.frombuffer(decompressed, dtype=np.int8)
        dequantized = quantized.astype(np.float32) / scale
        
        hidden_states = dequantized.reshape(1, shape1, shape2)
        
        return torch.tensor(hidden_states, device=self.device)


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


class ContinuousBatcher:
    """Continuous batching for maximum GPU utilization"""
    
    def __init__(self, max_batch_size: int = 64, timeout_ms: int = 10):
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.pending_requests = asyncio.Queue()
        self.processing = False
        
    async def add_request(self, request_data: Dict) -> Dict:
        future = asyncio.Future()
        await self.pending_requests.put((request_data, future))
        return await future
    
    async def process_batches(self, process_fn):
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
                    results = await process_fn(batch)
                    for future, result in zip(futures, results):
                        future.set_result(result)
                except Exception as e:
                    for future in futures:
                        future.set_exception(e)


class DecoderService:
    """Main decoder service with FastAPI integration"""
    
    def __init__(self, model_path: Optional[str] = None, vocab_dir: str = "vocabs"):
        self.model_path = model_path
        self.vocab_dir = vocab_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[OptimizedUniversalDecoder] = None
        self.vocabulary_manager = VocabularyManager(vocab_dir)
        self.batcher = ContinuousBatcher()
        
        # Configuration
        self.jwt_secret = os.environ.get("DECODER_JWT_SECRET", "")
        if not self.jwt_secret:
            # Generate a secure random key if none is provided
            import secrets
            self.jwt_secret = secrets.token_hex(32)
            logger.warning("No JWT secret key provided, generated a random one. This will not persist across restarts.")
            
        self.model_version = os.environ.get("MODEL_VERSION", "1.0.0")
        
        # Create FastAPI app
        self.app = self._create_app()
        
    def _create_app(self) -> FastAPI:
        """Create FastAPI application"""
        app = FastAPI(
            title="Universal Decoder API",
            version=self.model_version,
            description="High-performance translation decoder service"
        )
        
        security = HTTPBearer()
        
        def require_jwt(credentials: HTTPAuthorizationCredentials = Depends(security)):
            token = credentials.credentials
            try:
                jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            except Exception:
                raise HTTPException(status_code=401, detail="Invalid token")
        
        @app.on_event("startup")
        async def startup():
            await self._initialize()
            asyncio.create_task(self.batcher.process_batches(self._process_batch_gpu))
            
        @app.get("/health")
        async def health():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "device": str(self.device),
                "model_loaded": self.model is not None
            }
        
        @app.get("/status")
        async def status():
            """Status endpoint with model info"""
            return {
                "model_version": self.model_version,
                "healthy": True,
                "device": str(self.device),
                "gpu_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            }
        
        @app.post("/decode")
        async def decode(request: Request, x_target_language: str = Header(None)):
            """Decode encoder output to target language"""
            try:
                compressed_data = await request.body()
                
                result = await self.batcher.add_request({
                    'compressed_data': compressed_data,
                    'target_lang': x_target_language
                })
                
                return result
                
            except Exception as e:
                logger.error(f"Decode error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/admin/reload_model", dependencies=[Depends(require_jwt)])
        async def reload_model():
            """Reload model (requires authentication)"""
            try:
                await self._load_model()
                return {"status": "Model reloaded successfully"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        return app
    
    async def _initialize(self):
        """Initialize the decoder service"""
        await self._load_model()
        logger.info(f"✅ Decoder ready on {self.device}")
        if torch.cuda.is_available():
            logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    async def _load_model(self):
        """Load or reload the model"""
        if self.model_path and os.path.exists(self.model_path):
            self.model = torch.load(self.model_path, map_location=self.device)
        else:
            self.model = OptimizedUniversalDecoder().to(self.device)
        
        self.model.eval()
        
        # Compile with torch.compile for faster inference
        if torch.__version__ >= "2.0.0":
            self.model = torch.compile(self.model, mode="reduce-overhead")
    
    async def _process_batch_gpu(self, batch: List[Dict]) -> List[Dict]:
        """Process batch on GPU with maximum efficiency"""
        
        with torch.cuda.amp.autocast():  # Mixed precision for speed
            # Prepare batch tensors
            encoder_outputs = []
            encoder_masks = []
            target_langs = []
            
            for item in batch:
                # Decompress encoder output
                decompressed = self._decompress_encoder_output(item['compressed_data'])
                
                encoder_outputs.append(decompressed['hidden_states'])
                encoder_masks.append(decompressed['attention_mask'])
                target_langs.append(item['target_lang'])
            
            # Stack tensors
            encoder_hidden = torch.tensor(np.stack(encoder_outputs)).to(self.device)
            encoder_mask = torch.tensor(np.stack(encoder_masks)).to(self.device)
            
            # Get target language IDs
            target_lang_ids = [
                self.vocabulary_manager.language_to_pack.get(lang, 3) 
                for lang in target_langs
            ]
            
            # Generate translations
            with torch.no_grad():
                output_ids, scores = self.model.generate(
                    encoder_hidden,
                    encoder_mask,
                    target_lang_ids[0],
                    max_length=128,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9
                )
            
            # Decode to text
            results = []
            for i, output in enumerate(output_ids):
                # Get vocabulary pack for target language
                vocab_pack = self.vocabulary_manager.get_vocab_for_pair('en', target_langs[i])
                
                # Decode tokens to text
                tokens = output.cpu().numpy()
                text = self._decode_tokens_to_text(tokens, vocab_pack)
                
                results.append({
                    'translation': text,
                    'target_lang': target_langs[i],
                    'scores': scores[i] if i < len(scores) else None
                })
            
            return results
    
    def _decompress_encoder_output(self, compressed_data: bytes) -> Dict:
        """Decompress encoder output"""
        if isinstance(compressed_data, bytes):
            metadata_size = 12
            if len(compressed_data) < metadata_size:
                raise ValueError("Invalid compressed data")
            
            shape1 = int.from_bytes(compressed_data[0:4], 'little')
            shape2 = int.from_bytes(compressed_data[4:8], 'little')
            scale = np.frombuffer(compressed_data[8:12], dtype=np.float32)[0]
            
            compressed = compressed_data[metadata_size:]
            decompressed = lz4.frame.decompress(compressed)
            
            quantized = np.frombuffer(decompressed, dtype=np.int8)
            dequantized = quantized.astype(np.float32) / scale
            
            hidden_states = dequantized.reshape(1, shape1, shape2)
            
            return {
                'hidden_states': hidden_states,
                'attention_mask': np.ones((1, shape1), dtype=np.int32)
            }
        else:
            return compressed_data
    
    def _decode_tokens_to_text(self, tokens: np.ndarray, vocab_pack) -> str:
        """Decode token IDs to text using vocabulary pack"""
        # Use cached id_to_token mapping from vocabulary pack
        id_to_token = vocab_pack.id_to_token
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
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
        """Run the decoder service"""
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            workers=workers,
            log_level="info"
        )


def create_decoder_service(model_path: Optional[str] = None, vocab_dir: str = "vocabs") -> DecoderService:
    """Factory function to create decoder service"""
    return DecoderService(model_path, vocab_dir)