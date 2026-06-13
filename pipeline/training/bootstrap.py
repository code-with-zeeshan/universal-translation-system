from utils.common_utils import RuntimeDirectoryManager
# pipeline/training/bootstrap.py
import torch
import torch.nn as nn
from transformers import (
    AutoModel, AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoConfig
)
import warnings
from typing import Optional, Dict, Any
import logging
from utils.security import validate_model_source, safe_load_model
from utils.exceptions import TrainingError

# --- ADDED: Import Path for directory creation ---
from pathlib import Path

logger = logging.getLogger(__name__)

class PretrainedModelBootstrapper:
    """Bootstrap our models from existing pretrained models - Updated for 2024/2025"""
    
    def __init__(self, device: str = "auto"):
        self.device = self._get_device(device)
        
    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device with proper error handling"""
        if device == "auto":
            if torch.cuda.is_available():
                # Check if CUDA is actually functional
                try:
                   torch.cuda.init()
                   return torch.device("cuda")
                except Exception as e:
                    logger.warning(f"CUDA available but initialization failed: {e}")

            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                try:
                    # Test MPS functionality
                    test_tensor = torch.ones(1).to('mps')
                    return torch.device("mps")
                except Exception as e:
                    logger.warning(f"MPS available but test failed: {e}")
            
            return torch.device("cpu")
        
        # Validate requested device
        try:
            torch.zeros(1).to(device)
            return torch.device(device)
        except Exception as e:
            logger.warning(f"Requested device '{device}' not available: {e}")
            return torch.device("cpu")
    
    def create_encoder_from_pretrained(self, 
                                    model_name: str = 'xlm-roberta-base',
                                    output_path: str = 'self.runtime_dirs.encoder_models_dir / "universal_encoder_initial.pt"',
                                    target_hidden_dim: int = 768) -> nn.Module:
        """Create encoder using modern AutoModel patterns with dimension adaptation"""
        
        logger.info(f"🔄 Loading {model_name} with AutoModel...")
        
        # Use safe loading with security validation
        try:
            if not validate_model_source(model_name):
                raise TrainingError(f"Untrusted model source: {model_name}")

            model = safe_load_model(
                model_name,
                model_class=AutoModel,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,  # Use fast tokenizers
                trust_remote_code=False
            )
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
        
        logger.info("📦 Extracting components with proper error handling...")
        
        # Extract embeddings with proper dimension handling
        try:
            pretrained_embeddings = model.embeddings.word_embeddings.weight.data
            vocab_size = tokenizer.vocab_size
            hidden_size = model.config.hidden_size
            source_hidden_size = hidden_size
            
            logger.info(f"  - Source embeddings: {pretrained_embeddings.shape}")
            logger.info(f"  - Vocab size: {vocab_size}")
            logger.info(f"  - Source hidden size: {source_hidden_size}")
            logger.info(f"  - Target hidden size: {target_hidden_dim}")
            
        except AttributeError as e:
            logger.error(f"❌ Error extracting embeddings: {e}")
            raise

        # Adapt embeddings if dimensions don't match
        if source_hidden_size != target_hidden_dim:
            logger.info(f"📏 Adapting embeddings from {source_hidden_size} to {target_hidden_dim}")
            pretrained_embeddings = self._adapt_pretrained_embeddings(
                pretrained_embeddings, source_hidden_size, target_hidden_dim
            )
        
        # Create encoder with proper initialization
        from runtime.encoder.universal_encoder import UniversalEncoder
        
        our_encoder = UniversalEncoder(
            max_vocab_size=min(32000, vocab_size),
            hidden_dim=target_hidden_dim,
            num_layers=6,
            num_heads=8,
            dropout=0.1,
            max_positions=64,
            device=self.device
        )
        
        # Copy weights with proper error handling and dimension checks
        logger.info("💉 Transferring pretrained weights...")
        
        with torch.no_grad():
            # Copy embeddings — keep original encoder's vocab size, copy what fits
            if hasattr(our_encoder, 'embedding_layer'):
                n_copy = min(our_encoder.embedding_layer.weight.size(0), pretrained_embeddings.size(0))
                our_encoder.embedding_layer.weight.data[:n_copy] = \
                    pretrained_embeddings[:n_copy].to(self.device)

            # Also try to copy transformer weights if dimensions match
            if source_hidden_size == target_hidden_dim and hasattr(model, 'encoder'):
                try:
                    # Copy layer weights where possible
                    for i, layer in enumerate(model.encoder.layer[:6]):  # First 6 layers
                        if hasattr(our_encoder, f'layer_{i}'):
                            our_layer = getattr(our_encoder, f'layer_{i}')
                            # Copy attention weights
                            if hasattr(layer, 'attention') and hasattr(our_layer, 'attention'):
                                our_layer.attention.load_state_dict(layer.attention.state_dict())
                            # Copy FFN weights
                            if hasattr(layer, 'intermediate') and hasattr(our_layer, 'intermediate'):
                                our_layer.intermediate.load_state_dict(layer.intermediate.state_dict())
                    logger.info("✅ Transferred transformer layer weights")
                except Exception as e:
                    logger.warning(f"⚠️  Could not transfer transformer weights: {e}")  
        
        # Save state_dict BEFORE torch.compile so keys don't get _orig_mod prefix
        checkpoint = {
            'model_state_dict': our_encoder.state_dict(),
            'config': {
                'base_model': model_name,
                'source_hidden_dim': source_hidden_size,
                'target_hidden_dim': target_hidden_dim,
                'num_layers': 6,
                'vocab_size': our_encoder.embedding_layer.weight.size(0),
                'device': str(self.device),
                'torch_version': torch.__version__,
                'transformers_version': __import__('transformers').__version__,
                'adaptation_applied': source_hidden_size != target_hidden_dim
            },
            'tokenizer_config': tokenizer.init_kwargs if hasattr(tokenizer, 'init_kwargs') else {}
        }

        # Apply torch.compile for better performance (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            logger.info("🚀 Applying torch.compile optimization...")
            our_encoder = torch.compile(our_encoder)
        
        logger.info("✅ Encoder initialized with modern practices")

        # --- ADDED: Ensure output directory exists ---
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(checkpoint, output_path)
        logger.info(f"💾 Saved encoder to {output_path}")
        
        return our_encoder

    def _adapt_pretrained_embeddings(self, pretrained_embeddings: torch.Tensor, 
                                source_dim: int, target_dim: int) -> torch.Tensor:
        """
        Adapt pretrained embeddings to different dimensions with multiple strategies
    
        Args:
            pretrained_embeddings: Original embeddings tensor
            source_dim: Original dimension
            target_dim: Target dimension
        
        Returns:
            Adapted embeddings tensor
        """
        if source_dim == target_dim:
            return pretrained_embeddings
    
        logger.info(f"Adapting embeddings from {source_dim} to {target_dim} dimensions")
    
        vocab_size = pretrained_embeddings.shape[0]
        device = pretrained_embeddings.device
        dtype = pretrained_embeddings.dtype
    
        if source_dim > target_dim:
            # Dimension reduction - use PCA-like approach
            logger.info("Using PCA-inspired dimension reduction")
        
            # Option 1: Linear projection (most common)
            projection = nn.Linear(source_dim, target_dim, bias=False)
            projection = projection.to(device=device, dtype=dtype)
        
            # Initialize with PCA-like weights if possible
            if vocab_size >= source_dim:
                # Compute covariance matrix on a subset
                subset_size = min(5000, vocab_size)
                subset_embeddings = pretrained_embeddings[:subset_size].float()
            
                # Center the embeddings
                mean = subset_embeddings.mean(dim=0, keepdim=True)
                centered = subset_embeddings - mean
            
                # Compute covariance
                cov = torch.mm(centered.t(), centered) / (subset_size - 1)
            
                # Get top eigenvectors
                try:
                    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
                    # Select top target_dim eigenvectors
                    top_eigenvectors = eigenvectors[:, -target_dim:]
                    projection.weight.data = top_eigenvectors.t().contiguous().to(device=device, dtype=dtype)
                except Exception as e:
                    logger.warning(f"PCA initialization failed: {e}, using Xavier init")
                    nn.init.xavier_uniform_(projection.weight)
            else:
                # Fallback to Xavier initialization
                nn.init.xavier_uniform_(projection.weight)
        
            # Project embeddings
            with torch.no_grad():
                adapted = projection(pretrained_embeddings.to(dtype=dtype))
            
                # Preserve relative norms
                original_norms = pretrained_embeddings.norm(dim=1, keepdim=True)
                adapted_norms = adapted.norm(dim=1, keepdim=True) + 1e-8 # Add epsilon to prevent division by zero
                scale = original_norms / adapted_norms
                adapted = adapted * scale
    
        else:
            # Dimension expansion - use structured initialization
            logger.info("Using structured dimension expansion")
        
            adapted = torch.zeros(vocab_size, target_dim, device=device, dtype=dtype)
        
            # Copy original dimensions
            adapted[:, :source_dim] = pretrained_embeddings
        
            # Initialize extra dimensions
            extra_dims = target_dim - source_dim
        
            # Strategy 1: Use linear combinations of existing dimensions
            if extra_dims <= source_dim:
                # Create mixing matrix
                mixing = torch.randn(extra_dims, source_dim, device=device) * 0.1
                extra_features = torch.mm(pretrained_embeddings, mixing.t())
                adapted[:, source_dim:] = extra_features
        
            # Strategy 2: Use noise with structure
            else:
                # Initialize with structured noise
                for i in range(source_dim, target_dim):
                    if i < 2 * source_dim:
                        # Mirror with noise
                        source_idx = i - source_dim
                        adapted[:, i] = pretrained_embeddings[:, source_idx] * 0.1 + \
                                    torch.randn(vocab_size, device=device) * 0.02
                    else:
                        # Random initialization with small magnitude
                        adapted[:, i] = torch.randn(vocab_size, device=device) * 0.02
        
            # Normalize to maintain stable training
            adapted = adapted / adapted.norm(dim=1, keepdim=True) * \
                    pretrained_embeddings.norm(dim=1, keepdim=True)
    
        logger.info(f"✅ Adapted embeddings shape: {adapted.shape}")
    
        # Validate adaptation
        if torch.isnan(adapted).any() or torch.isinf(adapted).any():
            logger.error("❌ Adaptation produced NaN or Inf values!")
            raise TrainingError("Invalid values in adapted embeddings")
    
        return adapted

    def create_decoder_from_mbart(self, 
                                model_name: str = 'facebook/mbart-large-50',
                                output_path: str = 'self.runtime_dirs.decoder_models_dir / "universal_decoder_initial.pt"',
                                encoder_dim: int = 512,
                                decoder_dim: int = 768,
                                max_seq_length: int = 512) -> nn.Module:
        """Create decoder using modern AutoModel patterns"""
        
        logger.info(f"🔄 Loading {model_name} with AutoModel...")
        
        try:
            # Use AutoModel instead of specific model class
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
                trust_remote_code=False
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                trust_remote_code=False
            )
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
        
        # Extract decoder with proper error handling
        try:
            decoder = model.model.decoder
            config = model.config
            
        except AttributeError as e:
            logger.error(f"❌ Error extracting decoder: {e}")
            raise
        
        logger.info("📦 Creating decoder with modern architecture...")
        
        from runtime.cloud_decoder import OptimizedUniversalDecoder
        
        our_decoder = OptimizedUniversalDecoder(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            num_layers=8,
            num_heads=12,
            vocab_size=min(32000, tokenizer.vocab_size),
            max_length=max_seq_length,
            device=self.device
        )
        
        # Transfer weights with proper dimension handling
        logger.info("💉 Transferring knowledge with dimension adaptation...")
        
        with torch.no_grad():
            # Transfer embeddings — copy what fits, adapt dims if needed
            if hasattr(decoder, 'embed_tokens') and hasattr(our_decoder, 'embedding'):
                pretrained_embeddings = decoder.embed_tokens.weight.data
                our_vocab_size = min(our_decoder.embedding.num_embeddings, pretrained_embeddings.size(0))
                
                if pretrained_embeddings.size(1) != our_decoder.decoder_dim:
                    logger.info(f"Adapting decoder embeddings from {pretrained_embeddings.size(1)} to {our_decoder.decoder_dim}")
                    adapted = self._adapt_pretrained_embeddings(
                        pretrained_embeddings[:our_vocab_size],
                        pretrained_embeddings.size(1),
                        our_decoder.decoder_dim
                    )
                    our_decoder.embedding.weight.data[:our_vocab_size] = adapted.to(self.device)
                else:
                    our_decoder.embedding.weight.data[:our_vocab_size] = pretrained_embeddings[:our_vocab_size].to(self.device)
    
            # Transfer decoder layers
            for i in range(min(len(decoder.layers), len(our_decoder.layers))):
                try:
                    our_decoder.layers[i].load_state_dict(decoder.layers[i].state_dict())
                except Exception as e:
                    logger.warning(f"Could not transfer layer {i}: {e}")
        
        # Save state_dict BEFORE torch.compile
        checkpoint = {
            'model_state_dict': our_decoder.state_dict(),
            'config': {
                'base_model': model_name,
                'encoder_dim': encoder_dim,
                'decoder_dim': decoder_dim,
                'num_layers': 8,
                'vocab_size': min(50000, tokenizer.vocab_size),
                'device': str(self.device),
                'torch_version': torch.__version__,
                'transformers_version': __import__('transformers').__version__
            }
        }

        # Apply modern optimizations
        if hasattr(torch, 'compile'):
            logger.info("🚀 Applying torch.compile optimization...")
            our_decoder = torch.compile(our_decoder)
        
        logger.info("✅ Decoder initialized with modern practices")
        
        # --- ADDED: Ensure output directory exists ---
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(checkpoint, output_path)
        return our_decoder
    
    def extract_vocabulary_from_nllb(self, 
                                   model_name: str = 'facebook/nllb-200-distilled-600M') -> tuple:
        """Extract vocabulary using modern tokenizer patterns"""
        
        logger.info(f"🔄 Loading {model_name} tokenizer...")
        
        try:
            # Use AutoTokenizer instead of specific tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                trust_remote_code=False
            )
            
        except Exception as e:
            logger.error(f"❌ Error loading tokenizer: {e}")
            raise
        
        # Define supported languages
        our_languages = {
            'eng_Latn', 'spa_Latn', 'fra_Latn', 'deu_Latn', 'zho_Hans',
            'jpn_Jpan', 'kor_Hang', 'arb_Arab', 'hin_Deva', 'rus_Cyrl',
            'por_Latn', 'ita_Latn', 'tur_Latn', 'tha_Thai',
            'pol_Latn', 'ukr_Cyrl', 'nld_Latn', 'ind_Latn', 'swe_Latn'
        }
        
        logger.info("📊 Extracting vocabulary with modern practices...")
        
        # Build vocabulary with proper error handling
        try:
            vocab = tokenizer.get_vocab()
            our_vocab = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
            nllb_to_ours = {}
            
            idx = 4
            for token, token_id in vocab.items():
                if self._is_relevant_token(token, our_languages) and idx < 50000:
                    our_vocab[token] = idx
                    nllb_to_ours[token_id] = idx
                    idx += 1
            
            logger.info(f"✅ Extracted {len(our_vocab)} tokens with modern tokenizer")
            
        except Exception as e:
            logger.error(f"❌ Error extracting vocabulary: {e}")
            raise
        
        return our_vocab, nllb_to_ours
    
    def _is_relevant_token(self, token: str, languages: set) -> bool:
        """Check if token is relevant for our languages"""
        # Implement proper token filtering logic
        # This is a simplified version
        return len(token) > 0 and not token.startswith('  ')

# Usage with modern practices
if __name__ == "__main__":
    # Initialize with automatic device selection
    bootstrapper = PretrainedModelBootstrapper(device="auto")
    
    try:
        # Create models with error handling
        encoder = bootstrapper.create_encoder_from_pretrained()
        decoder = bootstrapper.create_decoder_from_mbart()
        vocab, mapping = bootstrapper.extract_vocabulary_from_nllb()
        
        logger.info("🎉 All models bootstrapped successfully!")
        
    except Exception as e:
        logger.error(f"❌ Bootstrap failed: {e}")
        raise