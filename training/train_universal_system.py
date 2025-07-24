# training/train_universal_system.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import wandb
from tqdm import tqdm
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json
import time
from contextlib import contextmanager
import safetensors.torch
from vocabulary.vocabulary_manager import VocabularyManager
import yaml

GPU_CONFIG_MAP = [
    ("A100", "config/training_a100.yaml"),
    ("V100", "config/training_v100.yaml"),
    ("3090", "config/training_rtx3090.yaml"),
    ("T4", "config/training_t4.yaml"),
]

def auto_select_config():
    import torch
    if not torch.cuda.is_available():
        print("No GPU detected, using default config.")
        return "config/training_t4.yaml"  # fallback or CPU config
    gpu_name = torch.cuda.get_device_name(0)
    for key, config_path in GPU_CONFIG_MAP:
        if key in gpu_name:
            print(f"Detected GPU: {gpu_name}, using config: {config_path}")
            return config_path
    print(f"Unknown GPU: {gpu_name}, using default config.")
    return "config/training_t4.yaml"  # fallback

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

from memory_efficient_training import (
    MemoryOptimizedTrainer, 
    MemoryConfig, 
    DynamicBatchSizer,
    create_modern_training_setup
)

logger = logging.getLogger(__name__)

class ModernUniversalSystemTrainer:
    """Modern universal system trainer with all latest optimizations"""
    
    def __init__(self, encoder, decoder, train_data_path, val_data_path, 
                 config: MemoryConfig = None, experiment_name: str = "universal-translation"):
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.experiment_name = experiment_name
        
        # Initialize memory optimized trainer
        self.memory_config = config or MemoryConfig()
        self.trainer = MemoryOptimizedTrainer(encoder, self.memory_config)
        
        # Move models to device
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        
        # Setup optimized models
        self._setup_models()
        
        # Create datasets
        self.train_dataset = ModernParallelDataset(train_data_path)
        self.val_dataset = ModernParallelDataset(val_data_path)
        
        # Create modern optimizers
        self.encoder_optimizer = self.trainer.create_optimized_optimizer(
            self.encoder.parameters(), lr=5e-5, weight_decay=0.01
        )
        self.decoder_optimizer = self.trainer.create_optimized_optimizer(
            self.decoder.parameters(), lr=5e-5, weight_decay=0.01
        )
        
        # Dynamic batch sizing
        self.batch_sizer = DynamicBatchSizer(initial_batch_size=32, max_batch_size=128)
        
        # Initialize wandb with modern config
        self._setup_wandb()
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.memory_config.mixed_precision else None
        
        logger.info("🚀 Modern Universal System Trainer initialized!")
    
    def _setup_models(self):
        """Setup models with modern optimizations"""
        
        # Apply memory optimizations to both models
        if self.memory_config.compile_model:
            try:
                self.encoder = torch.compile(
                    self.encoder, 
                    mode=self.memory_config.compile_mode,
                    dynamic=True
                )
                self.decoder = torch.compile(
                    self.decoder, 
                    mode=self.memory_config.compile_mode,
                    dynamic=True
                )
                logger.info("✅ Models compiled successfully")
            except Exception as e:
                logger.warning(f"⚠️ Model compilation failed: {e}")
        
        # Enable channels last if configured
        if self.memory_config.use_channels_last:
            try:
                self.encoder = self.encoder.to(memory_format=torch.channels_last)
                self.decoder = self.decoder.to(memory_format=torch.channels_last)
                logger.info("✅ Channels last memory format enabled")
            except Exception as e:
                logger.warning(f"⚠️ Channels last format failed: {e}")
    
    def _setup_wandb(self):
        """Setup wandb with comprehensive config"""
        
        wandb_config = {
            "model": {
                "encoder_params": sum(p.numel() for p in self.encoder.parameters()),
                "decoder_params": sum(p.numel() for p in self.decoder.parameters()),
                "total_params": sum(p.numel() for p in self.encoder.parameters()) + 
                              sum(p.numel() for p in self.decoder.parameters())
            },
            "training": {
                "mixed_precision": self.memory_config.mixed_precision,
                "gradient_checkpointing": self.memory_config.gradient_checkpointing,
                "compile_model": self.memory_config.compile_model,
                "flash_attention": self.memory_config.use_flash_attention,
                "dtype": str(self.memory_config.dtype),
                "compile_mode": self.memory_config.compile_mode
            },
            "system": {
                "pytorch_version": torch.__version__,
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "device": str(self.device),
                "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None
            }
        }
        
        wandb.init(
            project="universal-translation", 
            name=f"{self.experiment_name}-{time.strftime('%Y%m%d-%H%M%S')}",
            config=wandb_config,
            tags=["modern", "memory-optimized", "universal-system"]
        )
    
    @contextmanager
    def training_mode(self):
        """Context manager for training mode"""
        self.encoder.train()
        self.decoder.train()
        try:
            yield
        finally:
            pass
    
    @contextmanager
    def validation_mode(self):
        """Context manager for validation mode"""
        self.encoder.eval()
        self.decoder.eval()
        try:
            yield
        finally:
            pass
    
    def train(self, num_epochs: int = 10, save_every: int = 1, 
              validate_every: int = 1, log_every: int = 100):
        """Complete modern training loop"""
        
        # Create optimized dataloaders
        current_batch_size = self.batch_sizer.current_batch_size
        train_loader = self.trainer.create_optimized_dataloader(
            self.train_dataset, 
            batch_size=current_batch_size,
            shuffle=True,
            num_workers=8
        )
        
        # Modern schedulers with warm restarts
        num_training_steps = len(train_loader) * num_epochs
        
        # Use modern PyTorch schedulers instead of transformers
        encoder_scheduler = CosineAnnealingWarmRestarts(
            self.encoder_optimizer,
            T_0=num_training_steps // 4,  # Restart every quarter of training
            T_mult=2,  # Double the restart period each time
            eta_min=1e-7
        )
        
        decoder_scheduler = CosineAnnealingWarmRestarts(
            self.decoder_optimizer,
            T_0=num_training_steps // 4,
            T_mult=2,
            eta_min=1e-7
        )
        
        logger.info(f"🚀 Starting training for {num_epochs} epochs")
        logger.info(f"📊 Training steps: {num_training_steps}, Batch size: {current_batch_size}")
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            with self.training_mode():
                epoch_loss = self._train_epoch(
                    train_loader, 
                    encoder_scheduler, 
                    decoder_scheduler,
                    epoch,
                    log_every
                )
            
            # Validation
            if epoch % validate_every == 0:
                val_loss = self._validate_epoch()
                
                # Log epoch results
                wandb.log({
                    'epoch/train_loss': epoch_loss,
                    'epoch/val_loss': val_loss,
                    'epoch/learning_rate': encoder_scheduler.get_last_lr()[0],
                    'epoch/time': time.time() - epoch_start_time,
                    'epoch/current_batch_size': current_batch_size
                })
                
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Train: {epoch_loss:.4f}, Val: {val_loss:.4f}")
                
                # Adaptive batch sizing based on validation loss
                if hasattr(self, 'prev_val_loss'):
                    if val_loss > self.prev_val_loss * 1.05:  # Loss increased by 5%
                        current_batch_size = self.batch_sizer.decrease_batch_size()
                        logger.info(f"📉 Reduced batch size to {current_batch_size}")
                    elif val_loss < self.prev_val_loss * 0.95:  # Loss decreased by 5%
                        current_batch_size = self.batch_sizer.increase_batch_size()
                        logger.info(f"📈 Increased batch size to {current_batch_size}")
                
                self.prev_val_loss = val_loss
            else:
                val_loss = float('inf')
            
            # Save checkpoint
            if epoch % save_every == 0:
                self._save_checkpoint(epoch, epoch_loss, val_loss)
            
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info("🎉 Training completed!")
        wandb.finish()
    
    def _train_epoch(self, train_loader, encoder_scheduler, decoder_scheduler, epoch, log_every):
        """Train one epoch with modern features"""
        
        epoch_loss = 0
        step_count = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            step_start_time = time.time()
            
            # Move batch to device with non-blocking transfer
            source_ids = batch['source_ids'].to(self.device, non_blocking=True)
            target_ids = batch['target_ids'].to(self.device, non_blocking=True)
            source_mask = batch['source_mask'].to(self.device, non_blocking=True)
            target_mask = batch['target_mask'].to(self.device, non_blocking=True)
            
            # Dynamic vocabulary loading
            vocab_pack = batch['vocab_pack']
            self.encoder.load_vocabulary_pack(vocab_pack)
            self.decoder.load_vocabulary_pack(vocab_pack)
            
            # Forward pass with mixed precision
            if self.memory_config.mixed_precision:
                with torch.cuda.amp.autocast():
                    loss = self._forward_pass(source_ids, target_ids, source_mask, vocab_pack)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping before unscaling
                self.scaler.unscale_(self.encoder_optimizer)
                self.scaler.unscale_(self.decoder_optimizer)
                
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.0)
                
                # Optimizer steps with scaling
                self.scaler.step(self.encoder_optimizer)
                self.scaler.step(self.decoder_optimizer)
                self.scaler.update()
            else:
                # Standard precision training
                loss = self._forward_pass(source_ids, target_ids, source_mask, vocab_pack)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.0)
                
                # Optimizer steps
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()
            
            # Scheduler steps
            encoder_scheduler.step()
            decoder_scheduler.step()
            
            # Zero gradients
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            
            # Update metrics
            epoch_loss += loss.item()
            step_count += 1
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{encoder_scheduler.get_last_lr()[0]:.2e}",
                'step_time': f"{time.time() - step_start_time:.2f}s"
            })
            
            # Logging
            if self.global_step % log_every == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/encoder_lr': encoder_scheduler.get_last_lr()[0],
                    'train/decoder_lr': decoder_scheduler.get_last_lr()[0],
                    'train/step_time': time.time() - step_start_time,
                    'train/global_step': self.global_step,
                    'system/gpu_memory': torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                })
        
        return epoch_loss / step_count
    
    def _forward_pass(self, source_ids, target_ids, source_mask, vocab_pack):
        """Forward pass with modern optimizations"""
        
        # Encoder forward pass
        encoder_output = self.encoder(source_ids, source_mask)
        
        # Decoder forward pass (teacher forcing)
        decoder_output = self.decoder(
            target_ids[:, :-1],
            encoder_output,
            encoder_attention_mask=source_mask
        )
        
        # Calculate loss
        loss = nn.functional.cross_entropy(
            decoder_output.reshape(-1, decoder_output.size(-1)),
            target_ids[:, 1:].reshape(-1),
            ignore_index=vocab_pack.tokens.get('<pad>', 0),
            label_smoothing=0.1  # Modern label smoothing
        )
        
        return loss
    
    def _validate_epoch(self):
        """Validation with modern inference mode"""
        
        val_loader = self.trainer.create_optimized_dataloader(
            self.val_dataset,
            batch_size=self.batch_sizer.current_batch_size * 2,  # Larger batch for validation
            shuffle=False,
            num_workers=4
        )
        
        total_loss = 0
        step_count = 0
        
        with self.validation_mode():
            with torch.inference_mode():  # Modern replacement for torch.no_grad()
                for batch in tqdm(val_loader, desc="Validating"):
                    # Move batch to device
                    source_ids = batch['source_ids'].to(self.device, non_blocking=True)
                    target_ids = batch['target_ids'].to(self.device, non_blocking=True)
                    source_mask = batch['source_mask'].to(self.device, non_blocking=True)
                    
                    # Dynamic vocabulary loading
                    vocab_pack = batch['vocab_pack']
                    self.encoder.load_vocabulary_pack(vocab_pack)
                    self.decoder.load_vocabulary_pack(vocab_pack)
                    
                    # Forward pass
                    if self.memory_config.mixed_precision:
                        with torch.cuda.amp.autocast():
                            loss = self._forward_pass(source_ids, target_ids, source_mask, vocab_pack)
                    else:
                        loss = self._forward_pass(source_ids, target_ids, source_mask, vocab_pack)
                    
                    total_loss += loss.item()
                    step_count += 1
        
        return total_loss / step_count if step_count > 0 else float('inf')
    
    def _save_checkpoint(self, epoch, train_loss, val_loss):
        """Save checkpoint with modern format and safetensors support"""
        
        checkpoint_data = {
            'epoch': epoch,
            'global_step': self.global_step,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'encoder_optimizer': self.encoder_optimizer.state_dict(),
            'decoder_optimizer': self.decoder_optimizer.state_dict(),
            'memory_config': self.memory_config.__dict__,
            'batch_sizer_state': self.batch_sizer.get_state(),
            'pytorch_version': torch.__version__,
            'timestamp': time.time()
        }
        
        # Save model weights separately using safetensors
        encoder_path = self.checkpoint_dir / f"encoder_epoch_{epoch}.safetensors"
        decoder_path = self.checkpoint_dir / f"decoder_epoch_{epoch}.safetensors"
        
        try:
            # Save with safetensors format
            safetensors.torch.save_file(
                self.encoder.state_dict(),
                encoder_path
            )
            safetensors.torch.save_file(
                self.decoder.state_dict(),
                decoder_path
            )
            
            # Save training state as regular pickle
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint_data, checkpoint_path)
            
            logger.info(f"💾 Checkpoint saved to {checkpoint_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ Safetensors save failed: {e}, falling back to torch.save")
            
            # Fallback to regular torch.save
            checkpoint_data.update({
                'encoder_state_dict': self.encoder.state_dict(),
                'decoder_state_dict': self.decoder.state_dict(),
            })
            
            fallback_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}_fallback.pt"
            torch.save(checkpoint_data, fallback_path)
        
        # Save best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            
            try:
                safetensors.torch.save_file(
                    self.encoder.state_dict(),
                    self.checkpoint_dir / "best_encoder.safetensors"
                )
                safetensors.torch.save_file(
                    self.decoder.state_dict(),
                    self.checkpoint_dir / "best_decoder.safetensors"
                )
                
                # Save best checkpoint metadata
                best_checkpoint = checkpoint_data.copy()
                best_checkpoint['is_best'] = True
                torch.save(best_checkpoint, self.checkpoint_dir / "best_checkpoint.pt")
                
                logger.info(f"🏆 New best model saved! Val loss: {val_loss:.4f}")
                
            except Exception as e:
                logger.warning(f"⚠️ Best model save failed: {e}")
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer_state: bool = True):
        """Load checkpoint with modern format support"""
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model states
        if 'encoder_state_dict' in checkpoint:
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        else:
            # Try to load from separate safetensors files
            encoder_path = checkpoint_path.parent / "encoder.safetensors"
            if encoder_path.exists():
                encoder_state = safetensors.torch.load_file(encoder_path)
                self.encoder.load_state_dict(encoder_state)
        
        if 'decoder_state_dict' in checkpoint:
            self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        else:
            decoder_path = checkpoint_path.parent / "decoder.safetensors"
            if decoder_path.exists():
                decoder_state = safetensors.torch.load_file(decoder_path)
                self.decoder.load_state_dict(decoder_state)
        
        # Load training state
        if load_optimizer_state:
            if 'encoder_optimizer' in checkpoint:
                self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
            if 'decoder_optimizer' in checkpoint:
                self.decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
        
        # Restore training state
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('val_loss', float('inf'))
        
        if 'batch_sizer_state' in checkpoint:
            self.batch_sizer.load_state(checkpoint['batch_sizer_state'])
        
        logger.info(f"✅ Checkpoint loaded from {checkpoint_path}")
        logger.info(f"📊 Resumed at global step {self.global_step}")


# Modern dataset class with improved features
class ModernParallelDataset(torch.utils.data.Dataset):
    """Modern parallel dataset with caching and preprocessing"""
    
    def __init__(self, data_path: str, cache_dir: Optional[str] =  None, vocab_dir: str = 'vocabs'):
        self.data_path = Path(data_path)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_path.parent / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load or create cached data
        self.data = self._load_or_create_cache()

        # Initialize VocabularyManager with vocab directory
        self.vocab_manager = VocabularyManager(vocab_dir=vocab_dir)
        
        logger.info(f"📚 Dataset loaded: {len(self.data)} samples")
    
    def _load_or_create_cache(self):
        """Load cached data or create cache from raw data"""
        
        cache_file = self.cache_dir / f"{self.data_path.stem}_cache.json"
        
        if cache_file.exists():
            logger.info(f"📦 Loading cached data from {cache_file}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        logger.info(f"🔄 Creating cache from {self.data_path}")
        data = self._load_raw_data()
        
        # Save cache
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return data
    
    def _load_raw_data(self):
        """Load and preprocess raw parallel data"""
        
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                try:
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        data.append({
                            'source': parts[0].strip(),
                            'target': parts[1].strip(),
                            'source_lang': parts[2].strip(),
                            'target_lang': parts[3].strip(),
                            'line_no': line_no
                        })
                except Exception as e:
                    logger.warning(f"⚠️ Error processing line {line_no}: {e}")
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get vocabulary pack for this language pair
        vocab_pack = self.vocab_manager.get_vocab_for_pair(
            item['source_lang'], 
            item['target_lang']
        )
        
        # Tokenize with modern preprocessing
        source_tokens = self._tokenize_with_special_tokens(
            item['source'], 
            item['source_lang'], 
            vocab_pack
        )
        target_tokens = self._tokenize_with_special_tokens(
            item['target'], 
            item['target_lang'], 
            vocab_pack
        )
        
        # Create attention masks
        source_mask = [1] * len(source_tokens)
        target_mask = [1] * len(target_tokens)
        
        return {
            'source_ids': torch.tensor(source_tokens, dtype=torch.long),
            'target_ids': torch.tensor(target_tokens, dtype=torch.long),
            'source_mask': torch.tensor(source_mask, dtype=torch.long),
            'target_mask': torch.tensor(target_mask, dtype=torch.long),
            'vocab_pack': vocab_pack,
            'metadata': {
                'source_lang': item['source_lang'],
                'target_lang': item['target_lang'],
                'line_no': item.get('line_no', idx)
            }
        }
    
    def _tokenize_with_special_tokens(self, text: str, lang: str, vocab_pack):
        """Tokenize text with proper special token handling"""
        
        # Add language-specific special tokens
        tokens = [vocab_pack.tokens.get('<bos>', 0)]
        
        # Simple tokenization (replace with proper tokenizer)
        word_tokens = text.lower().split()
        for word in word_tokens:
            token_id = vocab_pack.tokens.get(word, vocab_pack.tokens.get('<unk>', 1))
            tokens.append(token_id)
        
        tokens.append(vocab_pack.tokens.get('<eos>', 2))
        
        return tokens

# During training, simulate quantization
class QuantizationAwareTrainer:
    def train_step(self, model, input_data):
        # Normal forward pass
        output = model(input_data)
        
        # Simulate quantization effects
        if hasattr(self, 'quantization_aware') and self.quantization_aware:
            output = self.fake_quantize(output)
        
        return output
    
    def fake_quantize(self, tensor):
        # TODO: Implement fake quantization
        return tensor
        
        # This helps model learn to be robust to quantization!        


# Usage example
def main():
    """Main training function"""
    import torch
    # Auto-detect GPU and load config
    config_path = auto_select_config()
    config_dict = load_config(config_path)
    # You may want to convert config_dict to MemoryConfig, etc. as needed
    # For now, just print for confirmation
    print(f"Loaded config from: {config_path}")
    # Initialize models (assuming they're defined elsewhere)
    # encoder = YourEncoderModel()
    # decoder = YourDecoderModel()
    # Configure memory optimizations
    memory_config = MemoryConfig(
        mixed_precision=config_dict["training"].get("mixed_precision", True),
        gradient_checkpointing=config_dict["memory"].get("gradient_checkpointing", True),
        compile_model=config_dict["memory"].get("compile_model", True),
        compile_mode=config_dict["memory"].get("compile_mode", "reduce-overhead"),
        use_flash_attention=config_dict["memory"].get("use_flash_attention", True)
        # Add more fields as needed
    )
    # Initialize trainer
    trainer = ModernUniversalSystemTrainer(
        encoder=encoder,
        decoder=decoder,
        train_data_path="data/train.txt",
        val_data_path="data/val.txt",
        config=memory_config,
        experiment_name="universal-translation-v2"
    )
    # Start training
    trainer.train(
        num_epochs=20,
        save_every=2,
        validate_every=1,
        log_every=50
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()