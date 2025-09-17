# encoder/language_adapters.py
import torch
import torch.nn as nn
from typing import Dict, Optional, List
from pathlib import Path
import logging
from .adapter_composition import AdapterComposition

logger = logging.getLogger(__name__)

try:
    from safetensors.torch import save_file
except ImportError:
    save_file = None

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

        if hasattr(self.base_encoder, 'hidden_dim'):
            self.hidden_dim = self.base_encoder.hidden_dim
        else:
            self.hidden_dim = 1024  # Default
            logger.warning("Base encoder missing hidden_dim attribute, using default 1024")
        
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
        
        if save_file:
            save_file(self.language_adapters[language].state_dict(), output_path)
        else:
            logger.warning("safetensors not found, falling back to torch.save. It is recommended to install safetensors.")
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
        
        # Calculate size based on actual parameter sizes
        total_params = sum(p.numel() for p in quantized_model.parameters())
        size_mb = (total_params * next(quantized_model.parameters()).element_size()) / (1024 * 1024)
        
        # Save base model
        torch.save({
            'state_dict': quantized_model.state_dict(),
            'config': {
                'hidden_dim': self.hidden_dim,
                'num_layers': 6,
                'vocab_size': 50000,
                'quantization': quantization_mode,
                'model_size_mb': size_mb
            }
        }, model_path)
        
        logger.info(f"✅ Saved base model ({quantization_mode}): {size_mb:.1f}MB")

        # Save each adapter separately
        adapters_dir = output_dir / 'adapters'
        adapters_dir.mkdir(exist_ok=True)
        
        for lang, adapter in self.language_adapters.items():
            adapter_path = adapters_dir / f'{lang}_adapter.pt'
            torch.save(adapter.state_dict(), adapter_path)
            
            # Calculate adapter size
            adapter_params = sum(p.numel() for p in adapter.parameters())
            adapter_size_mb = (adapter_params * 4) / (1024 * 1024)
            logger.info(f"✅ Saved {lang} adapter: {adapter_size_mb:.2f}MB")

    def compose_adapters(
        self,
        source_adapter_name: str,
        target_adapter_name: str,
        composition_strategy: str = 'average',
        **kwargs
    ) -> str:
        """
        Create a composed adapter from two existing adapters using various strategies.

        Supported strategies and kwargs:
        - 'average': no kwargs
        - 'weighted': weights: List[float] of length 2
        - 'task_vector_add': base_adapter_name: str
        - 'task_vector_subtract': base_adapter_name: str
        - 'fisher': fisher_weights: List[Dict[str, Tensor]] for each adapter
        - 'regmean': lambda_reg: float
        - 'ties': threshold: float = 0.1, density: Optional[float] = None
        - 'dare': drop_rate: float = 0.5, rescale: bool = True
        - 'magnitude_pruning': sparsity: float = 0.5
        - 'linear_combination': coefficients: List[float] of length 2

        Returns the name of the composed adapter.
        """
        if source_adapter_name not in self.language_adapters or \
           target_adapter_name not in self.language_adapters:
            raise ValueError("Both source and target adapters must be loaded before composition.")

        composed_name = f"{source_adapter_name}->{target_adapter_name}_composed:{composition_strategy}"

        # Avoid re-creating if it already exists
        if composed_name in self.language_adapters:
            return composed_name

        source_adapter = self.language_adapters[source_adapter_name]
        target_adapter = self.language_adapters[target_adapter_name]

        # Dispatch to strategy
        if composition_strategy == 'average':
            composed_state_dict = AdapterComposition.average_weights([source_adapter, target_adapter])
        elif composition_strategy == 'weighted':
            weights = kwargs.get('weights')
            if not weights or len(weights) != 2:
                raise ValueError("'weighted' strategy requires weights=[w1, w2]")
            composed_state_dict = AdapterComposition.weighted_average([source_adapter, target_adapter], weights)
        elif composition_strategy in ('task_vector_add', 'task_vector_subtract'):
            base_name = kwargs.get('base_adapter_name')
            if not base_name or base_name not in self.language_adapters:
                raise ValueError("Provide valid base_adapter_name for task vector strategy")
            op = 'add' if composition_strategy == 'task_vector_add' else 'subtract'
            base = self.language_adapters[base_name]
            # Add target task vector and subtract source (or vice versa) depending on use-case
            composed_state_dict = AdapterComposition.task_vector_arithmetic(
                base_adapter=base,
                task_adapters=[source_adapter, target_adapter],
                scaling_factors=kwargs.get('scaling_factors', [1.0, 1.0]),
                operation=op
            )
        elif composition_strategy == 'fisher':
            fisher_weights = kwargs.get('fisher_weights')
            if not fisher_weights or len(fisher_weights) != 2:
                raise ValueError("'fisher' strategy requires fisher_weights for each adapter")
            composed_state_dict = AdapterComposition.fisher_weighted_average(
                [source_adapter, target_adapter], fisher_weights
            )
        elif composition_strategy == 'regmean':
            lambda_reg = float(kwargs.get('lambda_reg', 0.1))
            composed_state_dict = AdapterComposition.regmean([source_adapter, target_adapter], lambda_reg)
        elif composition_strategy == 'ties':
            threshold = float(kwargs.get('threshold', 0.1))
            density = kwargs.get('density', None)
            composed_state_dict = AdapterComposition.ties_merging([source_adapter, target_adapter], threshold, density)
        elif composition_strategy == 'dare':
            drop_rate = float(kwargs.get('drop_rate', 0.5))
            rescale = bool(kwargs.get('rescale', True))
            composed_state_dict = AdapterComposition.dare_merging([source_adapter, target_adapter], drop_rate, rescale)
        elif composition_strategy == 'magnitude_pruning':
            sparsity = float(kwargs.get('sparsity', 0.5))
            composed_state_dict = AdapterComposition.magnitude_pruning_merge([source_adapter, target_adapter], sparsity)
        elif composition_strategy == 'linear_combination':
            coefficients = kwargs.get('coefficients')
            if not coefficients or len(coefficients) != 2:
                raise ValueError("'linear_combination' requires coefficients=[c1, c2]")
            composed_state_dict = AdapterComposition.linear_combination([source_adapter, target_adapter], coefficients)
        else:
            raise NotImplementedError(f"Composition strategy '{composition_strategy}' not implemented.")

        # Create a new adapter and load the composed weights
        self.add_language_adapter(composed_name)
        self.language_adapters[composed_name].load_state_dict(composed_state_dict)

        logger.info(
            f"Created composed adapter '{composed_name}' using '{composition_strategy}' strategy."
        )
        return composed_name

    def compose_multiple_adapters(
        self,
        adapter_names: List[str],
        composition_strategy: str = 'average',
        composed_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Compose an arbitrary list of adapters. See compose_adapters for strategies/kwargs.
        """
        if not adapter_names or len(adapter_names) < 2:
            raise ValueError("Provide at least two adapter_names to compose.")
        missing = [n for n in adapter_names if n not in self.language_adapters]
        if missing:
            raise ValueError(f"Adapters not loaded: {missing}")

        modules = [self.language_adapters[n] for n in adapter_names]
        composed_name = composed_name or f"{'+' .join(adapter_names)}_composed:{composition_strategy}"
        if composed_name in self.language_adapters:
            return composed_name

        if composition_strategy == 'average':
            state = AdapterComposition.average_weights(modules)
        elif composition_strategy == 'weighted':
            weights = kwargs.get('weights')
            if not weights or len(weights) != len(modules):
                raise ValueError("'weighted' requires weights matching adapter count")
            state = AdapterComposition.weighted_average(modules, weights)
        elif composition_strategy == 'regmean':
            state = AdapterComposition.regmean(modules, float(kwargs.get('lambda_reg', 0.1)))
        elif composition_strategy == 'ties':
            state = AdapterComposition.ties_merging(modules, float(kwargs.get('threshold', 0.1)), kwargs.get('density'))
        elif composition_strategy == 'dare':
            state = AdapterComposition.dare_merging(modules, float(kwargs.get('drop_rate', 0.5)), bool(kwargs.get('rescale', True)))
        elif composition_strategy == 'magnitude_pruning':
            state = AdapterComposition.magnitude_pruning_merge(modules, float(kwargs.get('sparsity', 0.5)))
        elif composition_strategy == 'linear_combination':
            coeffs = kwargs.get('coefficients')
            if not coeffs or len(coeffs) != len(modules):
                raise ValueError("'linear_combination' requires coefficients matching adapter count")
            state = AdapterComposition.linear_combination(modules, coeffs)
        else:
            raise NotImplementedError(f"Composition strategy '{composition_strategy}' not implemented for multiple adapters.")

        self.add_language_adapter(composed_name)
        self.language_adapters[composed_name].load_state_dict(state)
        logger.info(f"Created composed adapter '{composed_name}' using '{composition_strategy}' strategy from {adapter_names}.")
        return composed_name