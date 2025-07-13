# cloud_decoder/optimized_decoder.py
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, Request, Header
from fastapi.responses import JSONResponse
import asyncio
from typing import List, Dict, Optional
import time
import triton_python_backend_utils as pb_utils
import msgpack
import lz4.frame

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
    ):
        super().__init__()
        
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        
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
    ) -> torch.Tensor:
        """Optimized forward pass with torch.jit support"""
        
        batch_size, seq_len = decoder_input_ids.shape
        device = decoder_input_ids.device
        
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
    def generate_optimized(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        target_lang_id: int,
        max_length: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Optimized generation with multiple decoding strategies
        """
        batch_size = encoder_hidden_states.size(0)
        device = encoder_hidden_states.device
        
        # Initialize with target language token
        decoder_input_ids = torch.full((batch_size, 1), target_lang_id, device=device)
        
        # Cache for key-value pairs (memory efficient)
        past_key_values = None
        
        for _ in range(max_length - 1):
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
            
            # Append to sequence
            decoder_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=1)
            
            # Check for end token
            if (next_tokens == 2).all():  # Assuming 2 is </s>
                break
        
        return decoder_input_ids


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
app = FastAPI(title="Universal Translation Decoder")

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
    
    print(f"✅ Decoder ready on {device}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")


@app.post("/decode")
async def decode(
    request: Request,
    x_target_language: str = Header(None)
):
    """
    Decode encoder output to target language
    """
    try:
        # Read compressed data
        compressed_data = await request.body()
        
        # Decompress
        encoder_output = decompress_encoder_output(compressed_data)
        
        # Add to batch
        result = await batcher.add_request({
            'encoder_output': encoder_output,
            'target_lang': x_target_language
        })
        
        return JSONResponse(content=result)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


async def process_batch_gpu(batch: List[Dict]) -> List[Dict]:
    """Process batch on GPU with maximum efficiency"""
    
    with torch.cuda.amp.autocast():  # Mixed precision for speed
        # Prepare batch tensors
        encoder_outputs = []
        encoder_masks = []
        target_langs = []
        
        for item in batch:
            encoder_outputs.append(item['encoder_output']['hidden_states'])
            encoder_masks.append(item['encoder_output']['attention_mask'])
            target_langs.append(item['target_lang'])
        
        # Stack tensors
        encoder_hidden = torch.tensor(np.stack(encoder_outputs)).to(device)
        encoder_mask = torch.tensor(np.stack(encoder_masks)).to(device)
        
        # Get target language IDs
        target_lang_ids = [vocabulary_manager.get_lang_id(lang) for lang in target_langs]
        
        # Generate translations
        with torch.no_grad():
            output_ids = model.generate_optimized(
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
            text = vocabulary_manager.decode(output.cpu().numpy(), target_langs[i])
            results.append({
                'translation': text,
                'target_lang': target_langs[i]
            })
        
        return results


def decompress_encoder_output(compressed_data: bytes) -> Dict:
    """Decompress encoder output"""
    # Read metadata
    metadata_size = 12
    shape1 = int.from_bytes(compressed_data[0:4], 'little')
    shape2 = int.from_bytes(compressed_data[4:8], 'little')
    scale = np.frombuffer(compressed_data[8:12], dtype=np.float32)[0]
    
    # Decompress data
    compressed = compressed_data[metadata_size:]
    decompressed = lz4.frame.decompress(compressed)
    
    # Dequantize
    quantized = np.frombuffer(decompressed, dtype=np.int8)
    dequantized = quantized.astype(np.float32) / scale
    
    # Reshape
    hidden_states = dequantized.reshape(1, shape1, shape2)
    
    return {
        'hidden_states': hidden_states,
        'attention_mask': np.ones((1, shape1), dtype=np.int32)
    }


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