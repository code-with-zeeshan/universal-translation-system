"""
LitServe-based decoder for universal translation.

Replaces FastAPI DecoderService with LitServe for 2x faster ML inference
via automatic continuous batching.

Key endpoints (built-in):
  POST /predict  — decode compressed encoder output (auto-batched)
  GET  /health   — health check
  GET  /metrics  — Prometheus metrics

Usage:
  python -m universal_decoder_node.litserve_decoder
  # or
  litserve serve universal_decoder_node.litserve_decoder:decoder_api
"""
import os
import time
import json
import logging
import numpy as np
import lz4.frame
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import litserve as ls
from fastapi import HTTPException

from .decoder import OptimizedUniversalDecoder
from .vocabulary import VocabularyManager
from .utils.rate_limiter import RateLimiter
from .utils.memory_manager import MemoryManager
from .config import DecoderConfig

logger = logging.getLogger(__name__)

config = DecoderConfig()
rate_limiter = RateLimiter(requests_per_minute=60, requests_per_hour=1000)

DECODER_JWT_SECRET = os.environ.get("DECODER_JWT_SECRET", "")
if not DECODER_JWT_SECRET:
    import secrets
    DECODER_JWT_SECRET = secrets.token_hex(32)
    logger.warning("No JWT secret provided — generated random one (not persistent)")


class DecoderLitAPI(ls.LitAPI):
    """LitServe API for the universal translation decoder with auto-batching."""

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
        self._jwt_secret = DECODER_JWT_SECRET
        self.start_time = time.time()

    def setup(self, device):
        """Load model and vocabulary on worker startup."""
        self.device = torch.device(device if device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"DecoderLitAPI setup on device={self.device}")

        self.vocabulary_manager = VocabularyManager(self.vocab_dir)
        self.memory_manager = MemoryManager.get_instance()

        if self.model_path and os.path.exists(self.model_path):
            self.model, _ = self._safe_load(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
        else:
            self.model = OptimizedUniversalDecoder(
                encoder_dim=getattr(config.model, 'encoder_dim', 1024),
                decoder_dim=getattr(config.model, 'decoder_dim', 512),
                num_layers=getattr(config.model, 'num_layers', 6),
                num_heads=getattr(config.model, 'num_heads', 8),
                vocab_size=getattr(config.model, 'vocab_size', 50000),
                max_length=getattr(config.model, 'max_length', 256),
            ).to(self.device)
            logger.info("Created fresh model (no checkpoint)")

        self.model.eval()
        self.model = self.memory_manager.optimize_for_inference(self.model)
        logger.info(f"Decoder ready — model size: {self._model_size_mb():.1f}MB")

    def _model_size_mb(self) -> float:
        if self.model is None:
            return 0.0
        total = sum(p.numel() * p.element_size() for p in self.model.parameters())
        return total / (1024 * 1024)

    def _safe_load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        model = OptimizedUniversalDecoder(
            encoder_dim=ckpt.get("encoder_dim", 1024),
            decoder_dim=ckpt.get("decoder_dim", 512),
            num_layers=ckpt.get("num_layers", 6),
            num_heads=ckpt.get("num_heads", 8),
            vocab_size=ckpt.get("vocab_size", 50000),
        ).to(self.device)
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd, strict=False)
        return model, ckpt.get("metadata", {})

    def _verify_jwt(self, token: str) -> bool:
        if not token:
            return False
        try:
            import jwt
            jwt.decode(token, self._jwt_secret, algorithms=["HS256"],
                       options={"require": ["exp", "iat", "nbf"]})
            return True
        except Exception:
            return False

    def decode_request(self, request) -> Tuple[np.ndarray, np.ndarray, str]:
        """Convert raw HTTP request → (hidden_states, attention_mask, target_lang).

        Handles: binary LZ4+int8 compressed body, auth via Authorization header,
        rate limiting, and X-Target-Language header.
        """
        client_id = request.headers.get("x-client-id", request.client.host if request.client else "unknown")
        if not rate_limiter.is_allowed(client_id):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        auth = request.headers.get("authorization", "")
        if auth and not self._verify_jwt(auth.replace("Bearer ", "")):
            raise HTTPException(status_code=401, detail="Invalid token")

        target_lang = request.headers.get("x-target-language", "en")
        compressed_data = request.body()

        if isinstance(compressed_data, bytes):
            if len(compressed_data) < 12:
                raise HTTPException(status_code=400, detail="Invalid compressed data")
            shape1 = int.from_bytes(compressed_data[0:4], 'little')
            shape2 = int.from_bytes(compressed_data[4:8], 'little')
            scale = np.frombuffer(compressed_data[8:12], dtype=np.float32)[0]
            compressed = compressed_data[12:]
            decompressed = lz4.frame.decompress(compressed)
            quantized = np.frombuffer(decompressed, dtype=np.int8)
            dequantized = quantized.astype(np.float32) / scale
            hidden_states = dequantized.reshape(1, shape1, shape2).astype(np.float32)
            attention_mask = np.ones((1, shape1), dtype=np.int32)
            return hidden_states, attention_mask, target_lang

        raise HTTPException(status_code=400, detail="Expected binary body")

    def batch(self, inputs):
        """Stack list of (hidden, mask, lang) into batched tensors."""
        hidden = np.concatenate([x[0] for x in inputs], axis=0)
        mask = np.concatenate([x[1] for x in inputs], axis=0)
        langs = [x[2] for x in inputs]
        return (
            torch.from_numpy(hidden).to(self.device),
            torch.from_numpy(mask).to(self.device),
            langs,
        )

    def predict(self, batched_input):
        """Run decoder.generate on batched encoder output → list of translations.

        LitServe auto-batches: when max_batch_size > 1, `predict` receives
        the batched output from `batch()` and must return a list.
        """
        hidden, mask, target_langs = batched_input
        target_lang_ids = [
            self.vocabulary_manager.language_to_pack.get(lang, 3)
            for lang in target_langs
        ]
        with torch.no_grad():
            output_ids, _ = self.model.generate(
                hidden, mask, target_lang_ids[0],
                max_length=128, temperature=0.7, top_k=50, top_p=0.9,
            )
        texts = []
        for i, tokens in enumerate(output_ids):
            vocab_pack = self.vocabulary_manager.get_vocab_for_pair('en', target_langs[i])
            id_to_token = vocab_pack.id_to_token
            text_tokens = []
            for tid in tokens.cpu().numpy():
                if tid == 2:
                    break
                if tid == 0:
                    continue
                text_tokens.append(id_to_token.get(tid, '<unk>'))
            text = ' '.join(text_tokens).replace(' ##', '').replace('\u2581', ' ').strip()
            texts.append(text)

        self._maybe_cleanup()
        return texts

    def encode_response(self, output: str) -> Dict[str, str]:
        return {"translation": output}

    def _maybe_cleanup(self):
        if np.random.random() < 0.05:
            stats = self.memory_manager.get_memory_stats()
            if stats.get('gpu_memory_percent', 0) > 80:
                self.memory_manager.cleanup()

    def health(self) -> bool:
        return self.model is not None


decoder_api = DecoderLitAPI()

def main():
    logging.basicConfig(level=logging.INFO)
    server = ls.LitServer(decoder_api, accelerator="auto")
    logger.info("Starting LitServe decoder on port 8000")
    server.run(port=8000)


if __name__ == "__main__":
    main()
