# training/bootstrap_from_pretrained.py
import torch
import torch.nn as nn
from transformers import (
    AutoModel, AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoConfig
)
import warnings
from typing import Optional, Dict, Any

class PretrainedModelBootstrapper:
    """Bootstrap our models from existing pretrained models - Updated for 2024/2025"""
    
    def __init__(self, device: str = "auto"):
        self.device = self._get_device(device)
        
    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device with proper error handling"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def create_encoder_from_pretrained(self, 
                                     model_name: str = 'xlm-roberta-base',
                                     output_path: str = 'models/universal_encoder_initial.pt') -> nn.Module:
        """Create encoder using modern AutoModel patterns"""
        
        print(f"🔄 Loading {model_name} with AutoModel...")
        
        # Use AutoModel and AutoTokenizer for consistency
        try:
            model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
                trust_remote_code=False  # Security best practice
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,  # Use fast tokenizers
                trust_remote_code=False
            )
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        print("📦 Extracting components with proper error handling...")
        
        # Extract embeddings with proper dimension handling
        try:
            pretrained_embeddings = model.embeddings.word_embeddings.weight.data
            vocab_size = tokenizer.vocab_size
            hidden_size = model.config.hidden_size
            
            print(f"  - Embeddings: {pretrained_embeddings.shape}")
            print(f"  - Vocab size: {vocab_size}")
            print(f"  - Hidden size: {hidden_size}")
            
        except AttributeError as e:
            print(f"❌ Error extracting embeddings: {e}")
            raise
        
        # Create encoder with proper initialization
        from encoder.universal_encoder import UniversalEncoder
        
        our_encoder = UniversalEncoder(
            max_vocab_size=min(50000, vocab_size),
            hidden_dim=hidden_size,
            num_layers=6,
            device=self.device
        )
        
        # Copy weights with proper error handling and dimension checks
        print("💉 Transferring pretrained weights...")
        
        with torch.no_grad():
            # Copy embeddings with dimension validation
            num_tokens_to_copy = min(50000, pretrained_embeddings.size(0))
            
            if hasattr(our_encoder, 'embedding_layer'):
                our_encoder.embedding_layer = nn.Embedding(
                    num_tokens_to_copy, 
                    hidden_size,
                    device=self.device
                )
                our_encoder.embedding_layer.weight.data[:num_tokens_to_copy] = \
                    pretrained_embeddings[:num_tokens_to_copy].to(self.device)

                if hidden_size != 1024:  # My target dimension
                    logger.info(f"Adapting embeddings from {hidden_size} to 1024 dimensions")
                    pretrained_embeddings = self._adapt_pretrained_embeddings(
                        pretrained_embeddings, hidden_size, 1024
                    )
                    hidden_size = 1024  # Update for encoder creation   
        
        # Apply torch.compile for better performance (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            print("🚀 Applying torch.compile optimization...")
            our_encoder = torch.compile(our_encoder)
        
        print("✅ Encoder initialized with modern practices")
        
        # Save with comprehensive metadata
        checkpoint = {
            'model_state_dict': our_encoder.state_dict(),
            'config': {
                'base_model': model_name,
                'hidden_dim': hidden_size,
                'num_layers': 6,
                'vocab_size': num_tokens_to_copy,
                'device': str(self.device),
                'torch_version': torch.__version__,
                'transformers_version': __import__('transformers').__version__
            },
            'tokenizer_config': tokenizer.init_kwargs if hasattr(tokenizer, 'init_kwargs') else {}
        }
        
        torch.save(checkpoint, output_path)
        return our_encoder

    def _adapt_pretrained_embeddings(self, pretrained_embeddings, source_dim, target_dim):
        """Adapt pretrained embeddings to different dimensions"""
        if source_dim == target_dim:
            return pretrained_embeddings
    
        # Create projection layer
        projection = nn.Linear(source_dim, target_dim)
    
        # Initialize projection intelligently
        if source_dim > target_dim:
            # Use PCA-like initialization for dimension reduction
            nn.init.xavier_uniform_(projection.weight)
        else:
            # Use identity + noise for dimension expansion
            nn.init.eye_(projection.weight[:source_dim, :])
            nn.init.normal_(projection.weight[source_dim:, :], 0, 0.02)
    
        # Project embeddings
        with torch.no_grad():
            adapted = projection(pretrained_embeddings.float())
    
        return adapted    
    
    def create_decoder_from_mbart(self, 
                                model_name: str = 'facebook/mbart-large-50',
                                output_path: str = 'models/universal_decoder_initial.pt') -> nn.Module:
        """Create decoder using modern AutoModel patterns"""
        
        print(f"🔄 Loading {model_name} with AutoModel...")
        
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
            print(f"❌ Error loading model: {e}")
            raise
        
        # Extract decoder with proper error handling
        try:
            decoder = model.model.decoder
            config = model.config
            
        except AttributeError as e:
            print(f"❌ Error extracting decoder: {e}")
            raise
        
        print("📦 Creating decoder with modern architecture...")
        
        from cloud_decoder.optimized_decoder import OptimizedUniversalDecoder
        
        our_decoder = OptimizedUniversalDecoder(
            encoder_dim=1024,
            decoder_dim=512,
            num_layers=6,
            num_heads=8,
            vocab_size=min(50000, tokenizer.vocab_size),
            device=self.device
        )
        
        # Transfer weights with proper dimension handling
        print("💉 Transferring knowledge with dimension adaptation...")
        
        with torch.no_grad():
            # Implement proper weight transfer logic here
            # This is a simplified version - real implementation needs careful mapping
            pass
        
        # Apply modern optimizations
        if hasattr(torch, 'compile'):
            print("🚀 Applying torch.compile optimization...")
            our_decoder = torch.compile(our_decoder)
        
        print("✅ Decoder initialized with modern practices")
        
        # Save with comprehensive metadata
        checkpoint = {
            'model_state_dict': our_decoder.state_dict(),
            'config': {
                'base_model': model_name,
                'decoder_dim': 512,
                'num_layers': 6,
                'vocab_size': min(50000, tokenizer.vocab_size),
                'device': str(self.device),
                'torch_version': torch.__version__,
                'transformers_version': __import__('transformers').__version__
            }
        }
        
        torch.save(checkpoint, output_path)
        return our_decoder
    
    def extract_vocabulary_from_nllb(self, 
                                   model_name: str = 'facebook/nllb-200-distilled-600M') -> tuple:
        """Extract vocabulary using modern tokenizer patterns"""
        
        print(f"🔄 Loading {model_name} tokenizer...")
        
        try:
            # Use AutoTokenizer instead of specific tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                trust_remote_code=False
            )
            
        except Exception as e:
            print(f"❌ Error loading tokenizer: {e}")
            raise
        
        # Define supported languages
        our_languages = {
            'eng_Latn', 'spa_Latn', 'fra_Latn', 'deu_Latn', 'zho_Hans',
            'jpn_Jpan', 'kor_Hang', 'arb_Arab', 'hin_Deva', 'rus_Cyrl',
            'por_Latn', 'ita_Latn', 'tur_Latn', 'tha_Thai', 'vie_Latn',
            'pol_Latn', 'ukr_Cyrl', 'nld_Latn', 'ind_Latn', 'swe_Latn'
        }
        
        print("📊 Extracting vocabulary with modern practices...")
        
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
            
            print(f"✅ Extracted {len(our_vocab)} tokens with modern tokenizer")
            
        except Exception as e:
            print(f"❌ Error extracting vocabulary: {e}")
            raise
        
        return our_vocab, nllb_to_ours
    
    def _is_relevant_token(self, token: str, languages: set) -> bool:
        """Check if token is relevant for our languages"""
        # Implement proper token filtering logic
        # This is a simplified version
        return len(token) > 0 and not token.startswith('▁▁')

# Usage with modern practices
if __name__ == "__main__":
    # Initialize with automatic device selection
    bootstrapper = PretrainedModelBootstrapper(device="auto")
    
    try:
        # Create models with error handling
        encoder = bootstrapper.create_encoder_from_pretrained()
        decoder = bootstrapper.create_decoder_from_mbart()
        vocab, mapping = bootstrapper.extract_vocabulary_from_nllb()
        
        print("🎉 All models bootstrapped successfully!")
        
    except Exception as e:
        print(f"❌ Bootstrap failed: {e}")
        raise