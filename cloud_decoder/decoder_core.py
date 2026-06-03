# cloud_decoder/decoder_core.py
"""
Core decoder model classes only — no server dependencies.
This file is safe to import during training.
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional, Dict, Any, Tuple, List
import asyncio
import time


class OptimizedDecoderLayer(nn.Module):
    """Single decoder layer optimized for GPU efficiency"""

    def __init__(self, decoder_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            decoder_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(decoder_dim)

        self.cross_attn = nn.MultiheadAttention(
            decoder_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(decoder_dim)

        self.ffn = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim * 4),
            nn.SiLU(),
            nn.Linear(decoder_dim * 4, decoder_dim),
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
        residual = x
        x = self.self_attn_norm(x)
        x, _ = self.self_attn(x, x, x, attn_mask=causal_mask, is_causal=True)
        x = residual + self.dropout(x)

        residual = x
        x = self.cross_attn_norm(x)
        x, _ = self.cross_attn(
            x, encoder_hidden, encoder_hidden,
            key_padding_mask=~encoder_mask if encoder_mask is not None else None
        )
        x = residual + self.dropout(x)

        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)

        return x


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
        dropout: float = 0.1,
        device: torch.device = None,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.device = device or torch.device('cpu')
        self.dropout = dropout
        self._gradient_checkpointing = gradient_checkpointing

        self.embedding = nn.Embedding(vocab_size, decoder_dim)
        self.positional_embedding = nn.Embedding(max_length, decoder_dim)

        self.encoder_adapter = nn.Sequential(
            nn.Linear(encoder_dim, decoder_dim * 2),
            nn.GELU(),
            nn.Linear(decoder_dim * 2, decoder_dim),
        )

        self.layers = nn.ModuleList([
            OptimizedDecoderLayer(decoder_dim=decoder_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(decoder_dim)

        self.output_projection = nn.Linear(decoder_dim, vocab_size, bias=False)
        self.output_projection.weight = self.embedding.weight

        self.target_language_adapters = nn.ModuleDict()

        self.apply(self._init_weights)

    def gradient_checkpointing_enable(self):
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self._gradient_checkpointing = False

    def add_target_language_adapter(self, language: str):
        if language not in self.target_language_adapters:
            self.target_language_adapters[language] = nn.Sequential(
                nn.Linear(self.decoder_dim, self.decoder_dim // 8),
                nn.GELU(),
                nn.Linear(self.decoder_dim // 8, self.decoder_dim),
                nn.LayerNorm(self.decoder_dim),
            )

    def get_target_languages(self) -> List[str]:
        return list(self.target_language_adapters.keys())

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
        compressed_embeddings: Optional[Dict[str, Any]] = None,
        target_lang: Optional[str] = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, seq_len = decoder_input_ids.shape
        device = decoder_input_ids.device

        if compressed_embeddings is not None:
            encoder_hidden_states = self.decompress_embeddings(compressed_embeddings)

        x = self.embedding(decoder_input_ids)
        positions = torch.arange(seq_len, device=device).expand(batch_size, -1)
        x = x + self.positional_embedding(positions)

        encoder_hidden = self.encoder_adapter(encoder_hidden_states)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1,
        )

        if self.training and self._gradient_checkpointing and encoder_attention_mask is not None:
            for layer in self.layers:
                x = checkpoint(
                    layer, x, encoder_hidden, causal_mask, encoder_attention_mask,
                    use_reentrant=False,
                )
        else:
            for layer in self.layers:
                x = layer(x, encoder_hidden, causal_mask, encoder_attention_mask)

        if target_lang is not None and target_lang in self.target_language_adapters:
            x = self.target_language_adapters[target_lang](x)

        x = self.layer_norm(x)
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
        pad_token_id: int = 0,
    ) -> Tuple[torch.Tensor, List[float]]:
        batch_size = encoder_hidden_states.size(0)
        device = encoder_hidden_states.device

        decoder_input_ids = torch.full((batch_size, 1), target_lang_id, device=device)

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        scores = []

        for step in range(max_length - 1):
            logits = self.forward(
                decoder_input_ids, encoder_hidden_states, encoder_attention_mask
            )

            next_token_logits = logits[:, -1, :] / temperature

            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            probs = torch.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)

            finished = finished | (next_tokens.squeeze(-1) == eos_token_id)
            next_tokens[finished] = pad_token_id
            decoder_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=1)

            token_scores = torch.gather(probs, 1, next_tokens).squeeze(-1)
            scores.append(token_scores.cpu().tolist())

            if finished.all():
                break

        return decoder_input_ids, scores


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
        while True:
            batch = []
            futures = []

            start_time = time.time()
            while len(batch) < self.max_batch_size:
                try:
                    timeout = self.timeout_ms / 1000 - (time.time() - start_time)
                    if timeout <= 0:
                        break
                    request_data, future = await asyncio.wait_for(
                        self.pending_requests.get(), timeout=timeout
                    )
                    batch.append(request_data)
                    futures.append(future)
                except asyncio.TimeoutError:
                    break

            if batch:
                try:
                    from .optimized_decoder import process_batch_gpu
                    results = await process_batch_gpu(batch)
                    for future, result in zip(futures, results):
                        future.set_result(result)
                except Exception as e:
                    for future in futures:
                        future.set_exception(e)
