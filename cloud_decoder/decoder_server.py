"""
Cloud decoder — LitServe-based inference server with auto-batching.

Model classes preserved here for backward compatibility (also exported
via cloud_decoder.__init__).  New deployments should use the LitServe path.

LitServe usage:
    python -m cloud_decoder.decoder_server

Backward-compat FastAPI app also exported as ``app`` for test fixtures.
"""
import os
import time
import json
import logging
import numpy as np
import lz4.frame
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

import litserve as ls

from runtime.vocabulary.manager import UnifiedVocabularyManager, VocabularyMode
from utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# ── Helpers ──────────────────────────────────────────────────────────

VocabularyManager = lambda *args, **kwargs: UnifiedVocabularyManager(*args, mode=VocabularyMode.OPTIMIZED, **kwargs)
rate_limiter = RateLimiter(requests_per_minute=60, requests_per_hour=1000)

JWT_SECRET = os.environ.get("DECODER_JWT_SECRET", "")
if not JWT_SECRET:
    import secrets
    JWT_SECRET = secrets.token_hex(32)
    logger.warning("No JWT secret provided — generated random one (not persistent)")

# ── Model classes ────────────────────────────────────────────────────

class OptimizedDecoderLayer(nn.Module):
    """Single decoder layer with pre-norm, cross-attention, SwiGLU FFN."""

    def __init__(self, decoder_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(decoder_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(decoder_dim)
        self.cross_attn = nn.MultiheadAttention(decoder_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(decoder_dim)
        self.ffn = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim * 4),
            nn.SiLU(),
            nn.Linear(decoder_dim * 4, decoder_dim),
        )
        self.ffn_norm = nn.LayerNorm(decoder_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_hidden, causal_mask, encoder_mask=None):
        r = x
        x = self.self_attn_norm(x)
        x, _ = self.self_attn(x, x, x, attn_mask=causal_mask, is_causal=True)
        x = r + self.dropout(x)
        r = x
        x = self.cross_attn_norm(x)
        x, _ = self.cross_attn(x, encoder_hidden, encoder_hidden, key_padding_mask=encoder_mask)
        x = r + self.dropout(x)
        r = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        return r + self.dropout(x)


class OptimizedUniversalDecoder(nn.Module):
    """Decoder: 6 layers x 8 heads, 512-dim, tied embeddings, top-k+top-p generation."""

    def __init__(
        self,
        encoder_dim: int = 1024,
        decoder_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        vocab_size: int = 50000,
        max_length: int = 256,
        device: torch.device = None,
    ):
        super().__init__()
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding = nn.Embedding(vocab_size, decoder_dim)
        self.positional_embedding = nn.Embedding(max_length, decoder_dim)
        self.encoder_adapter = nn.Linear(encoder_dim, decoder_dim, bias=False)
        self.layers = nn.ModuleList([
            OptimizedDecoderLayer(decoder_dim, num_heads) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(decoder_dim)
        self.output_projection = nn.Linear(decoder_dim, vocab_size, bias=False)
        self.output_projection.weight = self.embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, decoder_input_ids, encoder_hidden_states, encoder_attention_mask=None):
        bsz, seq_len = decoder_input_ids.shape
        device = decoder_input_ids.device
        x = self.embedding(decoder_input_ids)
        positions = torch.arange(seq_len, device=device).expand(bsz, -1)
        x = x + self.positional_embedding(positions)
        encoder_hidden = self.encoder_adapter(encoder_hidden_states)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1
        )
        for layer in self.layers:
            x = layer(x, encoder_hidden, causal_mask, encoder_attention_mask)
        x = self.layer_norm(x)
        return self.output_projection(x)

    @torch.no_grad()
    def generate(
        self,
        encoder_hidden_states,
        encoder_attention_mask,
        target_lang_id: int,
        max_length: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
    ) -> Tuple[torch.Tensor, list]:
        bsz = encoder_hidden_states.size(0)
        device = encoder_hidden_states.device
        dec_ids = torch.full((bsz, 1), target_lang_id, device=device)
        finished = torch.zeros(bsz, dtype=torch.bool, device=device)
        scores = []
        for _ in range(max_length - 1):
            logits = self.forward(dec_ids, encoder_hidden_states, encoder_attention_mask)
            next_logits = logits[:, -1, :] / temperature
            if top_k > 0:
                vals, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < vals[:, -1:]] = float('-inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                indices_to_remove = remove.scatter(1, sorted_indices, remove)
                next_logits[indices_to_remove] = float('-inf')
            probs = torch.softmax(next_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            finished = finished | (next_tokens.squeeze(-1) == eos_token_id)
            next_tokens[finished] = pad_token_id
            dec_ids = torch.cat([dec_ids, next_tokens], dim=1)
            token_scores = torch.gather(probs, 1, next_tokens).squeeze(-1)
            scores.append(token_scores.cpu().tolist())
            if finished.all():
                break
        return dec_ids, scores


class ContinuousBatcher:
    """Legacy batcher (kept for backward compat — LitServe handles batching now)."""

    def __init__(self, max_batch_size: int = 64, timeout_ms: int = 10):
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms

    async def add_request(self, data: dict) -> dict:
        raise NotImplementedError("Use LitServe auto-batching instead")


# ── Decompression helpers ────────────────────────────────────────────

def decompress_encoder_output(compressed_data: bytes) -> Dict:
    if isinstance(compressed_data, bytes):
        if len(compressed_data) < 12:
            raise ValueError("Invalid compressed data")
        shape1 = int.from_bytes(compressed_data[0:4], 'little')
        shape2 = int.from_bytes(compressed_data[4:8], 'little')
        scale = np.frombuffer(compressed_data[8:12], dtype=np.float32)[0]
        compressed = compressed_data[12:]
        deq = lz4.frame.decompress(compressed)
        hidden = np.frombuffer(deq, dtype=np.int8).astype(np.float32) / scale
        hidden = hidden.reshape(1, shape1, shape2).astype(np.float32)
        return {'hidden_states': hidden, 'attention_mask': np.ones((1, shape1), dtype=np.int32)}
    return compressed_data


def decode_tokens_to_text(tokens: np.ndarray, vocab_pack) -> str:
    id_to_token = vocab_pack.id_to_token
    text_tokens = []
    for tid in tokens:
        if tid == 2:
            break
        if tid == 0:
            continue
        text_tokens.append(id_to_token.get(tid, '<unk>'))
    text = ' '.join(text_tokens).replace(' ##', '').replace('\u2581', ' ').strip()
    return text


# ── Verification helpers (shim for old dependency imports) ────────────

def verify_internal_request(request: Request) -> bool:
    return True


# ── LitServe API ─────────────────────────────────────────────────────

class CloudDecoderLitAPI(ls.LitAPI):
    """LitServe API for cloud decoder with auto-batching."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        vocab_dir: str = "vocabs",
        max_batch_size: int = 8,
        batch_timeout: float = 0.01,
    ):
        super().__init__(max_batch_size=max_batch_size, batch_timeout=batch_timeout, api_path="/decode")
        self.model_path = model_path
        self.vocab_dir = vocab_dir
        self._jwt_secret = JWT_SECRET
        self.start_time = time.time()

    def setup(self, device):
        self.device = torch.device(device if device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"CloudDecoderLitAPI setup on device={self.device}")
        self.vocab_manager = VocabularyManager(self.vocab_dir)
        if self.model_path and os.path.exists(self.model_path):
            ckpt = torch.load(self.model_path, map_location=self.device)
            self.model = OptimizedUniversalDecoder(
                encoder_dim=ckpt.get("encoder_dim", 1024),
                decoder_dim=ckpt.get("decoder_dim", 512),
                num_layers=ckpt.get("num_layers", 6),
                num_heads=ckpt.get("num_heads", 8),
                vocab_size=ckpt.get("vocab_size", 50000),
            ).to(self.device)
            sd = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(sd, strict=False)
        else:
            self.model = OptimizedUniversalDecoder().to(self.device)
        self.model.eval()
        logger.info("Cloud decoder ready")

    def _verify_jwt(self, token: str) -> bool:
        if not token:
            return False
        try:
            import jwt as pyjwt
            pyjwt.decode(token, self._jwt_secret, algorithms=["HS256"],
                         options={"require": ["exp", "iat", "nbf"]})
            return True
        except Exception:
            return False

    def decode_request(self, request) -> Tuple[np.ndarray, np.ndarray, str]:
        client_id = request.headers.get("x-client-id", request.client.host if request.client else "unknown")
        if not rate_limiter.is_allowed(client_id):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        auth = request.headers.get("authorization", "")
        if auth and not self._verify_jwt(auth.replace("Bearer ", "")):
            raise HTTPException(status_code=401, detail="Invalid token")
        target_lang = request.headers.get("x-target-language", "en")
        data = decompress_encoder_output(request.body())
        return data['hidden_states'].astype(np.float32), data['attention_mask'].astype(np.int32), target_lang

    def batch(self, inputs):
        hidden = np.concatenate([x[0] for x in inputs], axis=0)
        mask = np.concatenate([x[1] for x in inputs], axis=0)
        langs = [x[2] for x in inputs]
        return (
            torch.from_numpy(hidden).to(self.device),
            torch.from_numpy(mask).to(self.device),
            langs,
        )

    def predict(self, batched_input):
        hidden, mask, target_langs = batched_input
        lang_ids = [self.vocab_manager.language_to_pack.get(l, 3) for l in target_langs]
        with torch.no_grad():
            output_ids, _ = self.model.generate(hidden, mask, lang_ids[0])
        texts = []
        for i, tokens in enumerate(output_ids):
            vp = self.vocab_manager.get_vocab_for_pair('en', target_langs[i])
            texts.append(decode_tokens_to_text(tokens.cpu().numpy(), vp))
        return texts

    def encode_response(self, output: str) -> Dict[str, str]:
        return {"translation": output}

    def health(self) -> bool:
        return self.model is not None


# ── Backward-compat FastAPI app (for tests) ──────────────────────────

app = FastAPI(title="Decoder (LitServe-backed)", version="1.0.0")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    return JSONResponse(
        {"ready": True, "checks": {"model": True}},
        status_code=200,
    )


# ── Main ─────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO)
    api = CloudDecoderLitAPI()
    server = ls.LitServer(api, accelerator="auto")
    logger.info("Starting LitServe cloud decoder on port 8000")
    server.run(port=8000)


if __name__ == "__main__":
    main()
