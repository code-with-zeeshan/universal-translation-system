# encoder/universal_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import os
from opentelemetry import trace

# --- MODIFIED ---
# Import our new custom layers
from .custom_layers import RotaryEmbedding, CustomTransformerEncoderLayer

tracer = trace.get_tracer(__name__)
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.0.0")

class UniversalEncoder(nn.Module):
    """
    PyTorch implementation of the Universal Encoder for training
    --- UPDATED WITH RoPE and SwiGLU ---
    """
    
    @property
    def model_version(self):
        return MODEL_VERSION

    def __init__(
        self,
        max_vocab_size: int = 50000,
        hidden_dim: int = 1024,
        num_layers: int = 6,
        num_heads: int = 16,
        max_positions: int = 2048, # Increased for better RoPE generalization (eg. from 128 to 2024) 
        dropout: float = 0.1,
        device: torch.device = None
    ):
        with tracer.start_as_current_span("UniversalEncoder.__init__") as span:
            span.set_attribute("model_version", MODEL_VERSION)
            super().__init__()
            
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.max_vocab_size = max_vocab_size  # Store for compatibility
            self.device = device or torch.device('cpu')
            
            # Embeddings 
            self.embedding_layer = nn.Embedding(max_vocab_size, hidden_dim)

            # --- REMOVED ---
            # self.position_embedding = nn.Embedding(max_positions, hidden_dim)
        
            # +++ ADDED +++
            # Add the Rotary Embedding module
            self.rotary_embeddings = RotaryEmbedding(dim=hidden_dim, max_position_embeddings=max_positions)
            
            # --- REPLACED ---
            # Replace the standard TransformerEncoder with a ModuleList of our custom layers
            self.transformer_layers = nn.ModuleList([
                CustomTransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4, # Standard FFN hidden dim
                    dropout=dropout
                ) for _ in range(num_layers)
            ])
            
            # --- REMOVED ---
            # self.transformer = nn.TransformerEncoder(...)
            
            # --- RENAMED for clarity ---
            self.output_norm = nn.LayerNorm(hidden_dim)
            self.dropout = nn.Dropout(dropout)
            
            # For vocabulary management
            self.current_vocab = None

            # Language adapters
            self.adapters = nn.ModuleDict()  
            self.adapter_dim = 64
            
            # Quantization flag
            self.is_quantized = False
            
            # Initialize weights
            self.apply(self._init_weights)
            
    def _init_weights(self, module):
        """Initialize weights matching bootstrap expectations"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        language: Optional[str] = None
    ):
        with tracer.start_as_current_span("UniversalEncoder.forward") as span:
            span.set_attribute("model_version", MODEL_VERSION)
            """
            Forward pass matching decoder expectations

            Args:
                input_ids: Token IDs [batch_size, seq_len]
                attention_mask: Attention mask [batch_size, seq_len]
                language: Optional language code for adapter
            
            Returns:
                [batch_size, seq_len, 1024] matching decoder input
            """
            batch_size, seq_len = input_ids.shape
            
            # Token embeddings
            hidden_states = self.embedding_layer(input_ids)
            
            # --- REMOVED ---
            # The old position embedding logic is gone.
            
            # Apply dropout to the token embeddings
            hidden_states = self.dropout(hidden_states)
            
            # +++ ADDED +++
            # Get the rotary frequencies for the current sequence length
            freqs_cis = self.rotary_embeddings(seq_len)
            
            # Convert attention mask for transformer
            # PyTorch transformer expects True for masked positions
            src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
            
            # --- REPLACED ---
            # Pass through our custom transformer layers
            for layer in self.transformer_layers:
                hidden_states = layer(hidden_states, freqs_cis, src_key_padding_mask)
            
            # Final layer norm
            hidden_states = self.output_norm(hidden_states)

            # Apply adapter if available (maintains quality!)
            if language and language in self.adapters:
                # Ensure adapter is on the correct device
                self.adapters[language].to(hidden_states.device)
                hidden_states = self.adapters[language](hidden_states)
            
            return hidden_states  # [batch, seq, 1024]
    
    def load_vocabulary_pack(self, vocab_pack):
        with tracer.start_as_current_span("UniversalEncoder.load_vocabulary_pack") as span:
            span.set_attribute("model_version", MODEL_VERSION)
            """Dynamically load vocabulary embeddings"""
            self.current_vocab = vocab_pack
        
            # Resize embedding layer to match pack
            pack_size = len(vocab_pack.tokens)

            # Save device before creating new embedding
            device = self.embedding_layer.weight.device

            # Save old embeddings if needed
            old_embeddings = self.embedding_layer.weight.data.clone()

            # Create new embedding layer
            self.embedding_layer = nn.Embedding(pack_size, self.hidden_dim)
            self.embedding_layer.to(device)
        
            # Load pre-computed embeddings if available
            if hasattr(vocab_pack, 'embeddings') and vocab_pack.embeddings:
                # This maintains quality even with smaller vocab!
                if isinstance(vocab_pack.embeddings, dict):
                    # If embeddings are stored as dict
                    for token, token_id in vocab_pack.tokens.items():
                        if token in vocab_pack.embeddings:
                            self.embedding_layer.weight.data[token_id] = torch.tensor(
                                vocab_pack.embeddings[token]
                            )
                else:
                    # If embeddings are stored as tensor
                    self.embedding_layer.weight.data = torch.tensor(vocab_pack.embeddings)
            else:        
                # Initialize with random embeddings
                nn.init.normal_(self.embedding_layer.weight, mean=0.0, std=0.02)
                # Using logger from the global scope if available
                logger.warning(f"No pre-computed embeddings in vocab pack {vocab_pack.name}, using random initialization")
        
            # Re-apply quantization if model was quantized
            if self.is_quantized:
                """Crucial for quantized models"""
                self.embedding_layer = torch.quantization.quantize_dynamic(
                    self.embedding_layer,
                    qconfig_spec={torch.nn.Embedding}, 
                    dtype=torch.qint8
                )

    def add_language_adapter(self, language: str):
        """Add a language-specific adapter"""
        if language not in self.adapters:
            self.adapters[language] = nn.Sequential(
                nn.Linear(self.hidden_dim, self.adapter_dim),
                nn.ReLU(),
                nn.Linear(self.adapter_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim)
            )
            # Initialize adapter
            for module in self.adapters[language].modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def enable_quantization(self):
        """Mark model as quantized"""
        self.is_quantized = True