# encoder/language_adapters.py
import torch
import torch.nn as nn
from typing import Dict, Optional, List
from pathlib import Path

class LanguageAdapter(nn.Module):
    """Lightweight language-specific adapter (2MB each)"""
    
    def __init__(self, hidden_dim: int = 1024, adapter_dim: int = 64):
        super().__init__()
        # Bottleneck design: 1024 → 64 → 1024
        self.down_project = nn.Linear(hidden_dim, adapter_dim)
        self.up_project = nn.Linear(adapter_dim, hidden_dim)
        self.activation = nn.GELU()
        
        # LayerNorm for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Initialize with small values to start as identity
        nn.init.normal_(self.down_project.weight, std=0.01)
        nn.init.normal_(self.up_project.weight, std=0.01)
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Save input for residual
        residual = hidden_states
        
        # Bottleneck transformation
        hidden_states = self.down_project(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.up_project(hidden_states)
        
        # Residual connection and normalization
        return self.layer_norm(hidden_states + residual)

class AdapterUniversalEncoder(nn.Module):
    """Universal encoder with language adapters for edge deployment"""
    
    def __init__(self, base_encoder_path: Optional[str] = None):
        super().__init__()

        from encoder.universal_encoder import UniversalEncoder

        # Create full-size encoder (will be quantized later for edge)
        self.base_encoder = UniversalEncoder(
            max_vocab_size=50000,
            hidden_dim=1024,  # Full size as per your architecture
            num_layers=6,
            num_heads=16,
            max_positions=128
        )

        # Load pretrained weights if available
        if base_encoder_path and Path(base_encoder_path).exists():
            checkpoint = torch.load(base_encoder_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                self.base_encoder.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                self.base_encoder.load_state_dict(checkpoint, strict=False)
        
        # Initialize adapter dictionary
        self.language_adapters = nn.ModuleDict()
        
        # Adapter configuration
        self.adapter_dim = 64
        self.hidden_dim = 1024
        
        # Track loaded adapters
        self.loaded_adapters = set()
        
        # Freeze base model by default
        self.freeze_base_encoder()
        
    def freeze_base_encoder(self):
        """Freeze base encoder parameters"""
        for param in self.base_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_base_encoder(self):
        """Unfreeze base encoder parameters"""
        for param in self.base_encoder.parameters():
            param.requires_grad = True
    
    def add_language_adapter(self, language: str):
        """Add a new language adapter"""
        if language not in self.language_adapters:
            self.language_adapters[language] = LanguageAdapter(
                hidden_dim=self.hidden_dim,
                adapter_dim=self.adapter_dim
            )
            self.loaded_adapters.add(language)
    
    def load_language_adapter(self, language: str, adapter_path: Optional[str] = None):
        """Dynamically load language adapter"""
        if language in self.loaded_adapters and adapter_path is None:
            return  # Already loaded
        
        # Create adapter if not exists
        if language not in self.language_adapters:
            self.add_language_adapter(language)
        
        # Load weights if path provided
        if adapter_path and Path(adapter_path).exists():
            adapter_state = torch.load(adapter_path, map_location='cpu')
            self.language_adapters[language].load_state_dict(adapter_state)
        
        self.loaded_adapters.add(language)

    def save_language_adapter(self, language: str, output_path: str):
        """Save a specific language adapter"""
        if language not in self.language_adapters:
            raise ValueError(f"No adapter found for language: {language}")
        
        torch.save(self.language_adapters[language].state_dict(), output_path)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                language: Optional[str] = None) -> torch.Tensor:
        # Get base encoding
        hidden_states = self.base_encoder(input_ids, attention_mask)
        
        # Apply language adapter if available
        if language and language in self.language_adapters and language in self.loaded_adapters:
            # Move adapter to same device as input
            device = input_ids.device
            self.language_adapters[language].to(device)
            
            # Apply adapter
            hidden_states = self.language_adapters[language](hidden_states)
        
        return hidden_states
    
    def save_edge_model(self, output_dir: str, quantization_mode: str = 'int8'):
        """Save edge-optimized model with adapters"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Quantize model based on mode
        if quantization_mode == 'int8':
            quantized_model = torch.quantization.quantize_dynamic(
                self.base_encoder,
                qconfig_spec={torch.nn.Linear, torch.nn.Embedding},
                dtype=torch.qint8
            )
            model_path = output_dir / 'base_encoder_int8.pt'
        elif quantization_mode == 'fp16':
            quantized_model = self.base_encoder.half()
            model_path = output_dir / 'base_encoder_fp16.pt'
        else:
            quantized_model = self.base_encoder
            model_path = output_dir / 'base_encoder_fp32.pt'
        
        # Calculate size
        total_params = sum(p.numel() for p in quantized_model.parameters())
        size_mb = (total_params * 4) / (1024 * 1024)  # Assuming float32
        if quantization_mode == 'int8':
            size_mb = size_mb / 4  # INT8 is ~1/4 the size
        elif quantization_mode == 'fp16':
            size_mb = size_mb / 2  # FP16 is half the size
        
        # Save base model
        torch.save({
            'state_dict': quantized_model.state_dict(),
            'config': {
                'hidden_dim': self.hidden_dim,
                'num_layers': 6,
                'vocab_size': 50000,
                'quantization': quantization_mode,
                'model_size_mb': sum_mb
            }
        }, model_path)
        
        print(f"✅ Saved base model ({quantization_mode}): {size_mb:.1f}MB")

        # Save each adapter separately
        adapters_dir = output_dir / 'adapters'
        adapters_dir.mkdir(exist_ok=True)
        
        for lang, adapter in self.language_adapters.items():
            adapter_path = adapters_dir / f'{lang}_adapter.pt'
            torch.save(adapter.state_dict(), adapter_path)
            
            # Calculate adapter size
            adapter_params = sum(p.numel() for p in adapter.parameters())
            adapter_size_mb = (adapter_params * 4) / (1024 * 1024)
            print(f"✅ Saved {lang} adapter: {adapter_size_mb:.2f}MB")