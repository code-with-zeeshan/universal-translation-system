# encoder/universal_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
import os
import logging
from utils.tracing import get_tracer

# --- MODIFIED ---
# Import our new custom layers
from .custom_layers import RotaryEmbedding, CustomTransformerEncoderLayer

tracer = get_tracer(__name__)
logger = logging.getLogger(__name__)
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.0.0")

LANGUAGE_CODES: List[str] = [
    "en", "es", "fr", "de", "zh",
    "ja", "ko", "ar", "hi", "ru",
    "pt", "it", "tr", "th", "pl",
    "uk", "nl", "id", "sv", "vi",
]

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
        device: torch.device = None,
        num_languages: int = 20,
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

            # Language ID embedding: added to all token positions as a per-language bias
            self.language_embedding = nn.Embedding(num_languages, hidden_dim)
            self._lang_to_idx = {code: i for i, code in enumerate(LANGUAGE_CODES[:num_languages])}

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
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        language: Optional[str] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
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
            if input_ids is None and inputs_embeds is not None:
                batch_size, seq_len = inputs_embeds.shape[:2]
                hidden_states = self.dropout(inputs_embeds)
            else:
                batch_size, seq_len = input_ids.shape
                hidden_states = self.embedding_layer(input_ids)
                hidden_states = self.dropout(hidden_states)

            # Add per-token language ID bias so the encoder knows the source language
            if language is not None:
                if isinstance(language, str):
                    lang_ids = torch.full((batch_size,), self._lang_to_idx.get(language, 0),
                                         device=hidden_states.device, dtype=torch.long)
                else:
                    lang_ids = torch.tensor(
                        [self._lang_to_idx.get(l, 0) for l in language],
                        device=hidden_states.device, dtype=torch.long
                    )
                lang_bias = self.language_embedding(lang_ids).unsqueeze(1)  # [B, 1, H]
                hidden_states = hidden_states + lang_bias
            
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
    
    def load_vocabulary_pack(self, vocab_pack, quantize=False):
        """
        Load a vocabulary pack into the model.
    
        Args:
            vocab_pack: VocabularyPack instance
            quantize: Whether to quantize the embeddings
        """
        with tracer.start_as_current_span("UniversalEncoder.load_vocabulary_pack") as span:
            span.set_attribute("vocab_size", len(vocab_pack.tokens))
            span.set_attribute("quantize", quantize)
        
            # Get current device
            device = self.embedding_layer.weight.device
        
            # Get pack size
            pack_size = len(vocab_pack.tokens) + len(vocab_pack.special_tokens)
        
            # Create new embedding layer with correct size
            new_embedding_layer = nn.Embedding(pack_size, self.hidden_dim)
            new_embedding_layer.to(device)
        
            # Initialize embeddings
            if hasattr(vocab_pack, 'embeddings') and vocab_pack.embeddings is not None:
                # Use pre-trained embeddings
                if isinstance(vocab_pack.embeddings, dict):
                    # Initialize with zeros
                    nn.init.zeros_(new_embedding_layer.weight)
                
                    # Set embeddings for each token
                    for token, token_id in vocab_pack.tokens.items():
                        if token in vocab_pack.embeddings:
                            new_embedding_layer.weight.data[token_id] = torch.tensor(
                                vocab_pack.embeddings[token], device=device
                            )
                else:
                    # Use full embedding matrix
                    new_embedding_layer.weight.data = torch.tensor(
                        vocab_pack.embeddings, device=device
                    )
            else:
                # Initialize randomly
                nn.init.normal_(new_embedding_layer.weight, mean=0.0, std=0.02)
            
            # Quantize if requested
            if quantize:
                new_embedding_layer = torch.quantization.quantize_dynamic(
                    new_embedding_layer,
                    {nn.Embedding},
                    dtype=torch.qint8
                )
                self.is_quantized = True
            
            # Replace the embedding layer
            self.embedding_layer = new_embedding_layer
        
            # Update vocabulary info
            self.vocab_size = pack_size
            self.vocab_pack_name = getattr(vocab_pack, 'name', 'custom')
        
            # Force garbage collection to clean up old embeddings
            import gc
            gc.collect()
        
            logger.info(
                f"Loaded vocabulary pack with {pack_size} tokens "
                f"(quantized={quantize})"
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