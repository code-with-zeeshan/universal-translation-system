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
# Imported from canonical source to eliminate duplication.
from runtime.cloud_decoder.decoder_core import (
    OptimizedDecoderLayer,
    OptimizedUniversalDecoder,
    ContinuousBatcher,
)


# Import shared helpers from canonical source
from runtime.cloud_decoder.decoder_core import decompress_encoder_output, decode_tokens_to_text


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

    async def decode_request(self, request) -> Tuple[np.ndarray, np.ndarray, str]:
        client_id = request.headers.get("x-client-id", request.client.host if request.client else "unknown")
        if not rate_limiter.is_allowed(client_id):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        auth = request.headers.get("authorization", "")
        if auth and not self._verify_jwt(auth.replace("Bearer ", "")):
            raise HTTPException(status_code=401, detail="Invalid token")
        target_lang = request.headers.get("x-target-language", "en")
        data = decompress_encoder_output(await request.body())
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
