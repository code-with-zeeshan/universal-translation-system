# encoder/custom_layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

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
    Uses lightweight multi-head attention implemented here to allow
    applying RoPE before attention without relying on internal
    torch.nn.MultiheadAttention projections.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = (self.head_dim) ** -0.5

        # Projections for Q, K, V and output
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
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

        # 2. Project to Q, K, V (batch, seq, d_model)
        q_proj = self.q_proj(x)
        k_proj = self.k_proj(x)
        v_proj = self.v_proj(x)
        
        # 3. Apply Rotary Embeddings to Query and Key (before head split)
        cos, sin = freqs_cis
        q_embed, k_embed = apply_rotary_pos_emb(q_proj, k_proj, cos, sin)

        # 4. Split heads: (batch, nhead, seq, head_dim)
        B, S, _ = q_embed.shape
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, S, self.nhead, self.head_dim).transpose(1, 2).contiguous()
        q = split_heads(q_embed)
        k = split_heads(k_embed)
        v = split_heads(v_proj)

        # 5. Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, S, S)

        # Apply key padding mask if provided (True = masked)
        if src_key_padding_mask is not None:
            # mask shape: (B, 1, 1, S)
            mask = src_key_padding_mask[:, None, None, :]
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, v)  # (B, H, S, head_dim)

        # 6. Merge heads back: (B, S, D)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        attn_out = self.out_proj(attn_out)

        # 7. Add & Norm (residual connection)
        src = src + self.dropout1(attn_out)
        
        # --- Feed-Forward Network ---
        # 8. Apply FFN with another residual connection
        x = self.norm2(src)
        ffn_output = self.ffn(x)
        src = src + self.dropout2(ffn_output)
        
        return src