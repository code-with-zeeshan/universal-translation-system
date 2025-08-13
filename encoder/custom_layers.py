# encoder/custom_layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    This replaces the standard FFN in the transformer layer.
    """
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        # The hidden dimension is typically larger than the input dimension
        # A common choice is 2/3 * 4 * dim, but we'll keep it simple for now.
        # We'll use the standard 4*dim from the original transformer.
        ffn_hidden_dim = hidden_dim

        self.w1 = nn.Linear(dim, ffn_hidden_dim, bias=bias)
        self.w2 = nn.Linear(ffn_hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, ffn_hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The SwiGLU logic: F.silu(w1(x)) * w3(x)
        # This gating mechanism allows the network to control information flow.
        gate = self.w1(x)
        silu_gate = F.silu(gate)
        up_projection = self.w3(x)
        gated_output = silu_gate * up_projection
        
        # Final projection back to the original dimension
        return self.w2(gated_output)

class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embeddings (RoPE).
    This module pre-computes the rotary frequencies.
    """
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        # Pre-compute the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build the cosine and sine cache
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Register as buffers so they are moved to the correct device automatically
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len: int):
        # Return the cached frequencies for the given sequence length
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...],
        )

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Positional Embedding to the query and key tensors."""
    cos = cos.squeeze(1).squeeze(0)  # (seq_len, dim)
    sin = sin.squeeze(1).squeeze(0)  # (seq_len, dim)
    
    # Reshape q and k to separate heads
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed        

class CustomTransformerEncoderLayer(nn.Module):
    """
    A custom transformer encoder layer that integrates RoPE and SwiGLU.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Replace the standard FFN with our SwiGLU implementation
        self.ffn = SwiGLU(dim=d_model, hidden_dim=dim_feedforward)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, freqs_cis: Tuple[torch.Tensor, torch.Tensor], src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # --- Self-Attention with RoPE ---
        # 1. Apply LayerNorm first (Pre-Norm)
        x = self.norm1(src)
        
        # 2. Get Q, K, V from the input
        q_proj = self.self_attn.in_proj_q(x)
        k_proj = self.self_attn.in_proj_k(x)
        v_proj = self.self_attn.in_proj_v(x)
        
        # 3. Apply Rotary Embeddings to Query and Key
        cos, sin = freqs_cis
        q_embed, k_embed = apply_rotary_pos_emb(q_proj, k_proj, cos, sin)
        
        # 4. Perform attention
        attn_output, _ = self.self_attn(q_embed, k_embed, v_proj, key_padding_mask=src_key_padding_mask)
        
        # 5. Add & Norm (residual connection)
        src = src + self.dropout1(attn_output)
        
        # --- Feed-Forward Network ---
        # 6. Apply FFN with another residual connection
        x = self.norm2(src)
        ffn_output = self.ffn(x)
        src = src + self.dropout2(ffn_output)
        
        return src