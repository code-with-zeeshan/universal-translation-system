# training/progressive_training.py (tier-based)
import torch
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Optional
import json
import numpy as np
from pathlib import Path
import logging
from utils.gpu_utils import optimize_gpu_memory
from utils.shutdown_handler import GracefulShutdown
from utils.model_versioning import ModelVersion
from utils.resource_monitor import resource_monitor

# Initialize GPU optimization
optimize_gpu_memory()

logger = logging.getLogger(__name__)

class ProgressiveTrainingStrategy:
    """Train incrementally from easy to hard languages"""
    
    def __init__(self, 
                 encoder: torch.nn.Module,
                 decoder: torch.nn.Module,
                 train_dataset,
                 val_dataset,
                 config_path: str = 'data/config.yaml'):
        
        self.encoder = encoder
        self.decoder = decoder
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Load config
        from data.data_utils import ConfigManager
        self.config = ConfigManager.load_config(config_path)
        
        # Define language tiers based on data availability
        self.language_tiers = {
            'tier1': {
                'languages': ['en', 'es', 'fr', 'de'],
                'reason': 'High-resource Indo-European',
                'epochs': 10,
                'lr': 5e-4,
                'batch_size': 64
            },
            'tier2': {
                'languages': ['zh', 'ja', 'ru', 'pt', 'it'],
                'reason': 'Major languages, different scripts',
                'epochs': 8,
                'lr': 3e-4,
                'batch_size': 48
            },
            'tier3': {
                'languages': ['ar', 'hi', 'ko', 'nl', 'pl'],
                'reason': 'Medium-resource, diverse',
                'epochs': 6,
                'lr': 2e-4,
                'batch_size': 32
            },
            'tier4': {
                'languages': ['tr', 'th', 'vi', 'uk', 'id', 'sv'],
                'reason': 'Lower-resource languages',
                'epochs': 4,
                'lr': 1e-4,
                'batch_size': 24
            }
        }
        
        # Training statistics
        self.training_history = {
            'tier_losses': {},
            'tier_accuracies': {},
            'total_time': 0
        }
    
    def get_tier_indices(self, tier_languages: List[str]) -> List[int]:
        """Get dataset indices for specific languages"""
        indices = []
        
        for idx in range(len(self.train_dataset)):
            sample = self.train_dataset[idx]
            # Check if sample's languages are in current tier
            if (sample['metadata']['source_lang'] in tier_languages or
                sample['metadata']['target_lang'] in tier_languages):
                indices.append(idx)
        
        return indices
    
    def create_tier_dataloader(self, tier_name: str) -> DataLoader:
        """Create dataloader for specific tier"""
        tier_config = self.language_tiers[tier_name]
        tier_languages = tier_config['languages']
        
        # Get indices for this tier
        indices = self.get_tier_indices(tier_languages)
        
        # Create subset
        subset = Subset(self.train_dataset, indices)
        
        # Create dataloader
        dataloader = DataLoader(
            subset,
            batch_size=tier_config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"Created dataloader for {tier_name} with {len(indices)} samples")
        return dataloader
    
    def train_progressive(self, 
                         save_dir: str = 'checkpoints/progressive',
                         use_adapters: bool = True, shutdown_handler=None):
        """Main progressive training loop"""
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Get all languages trained so far
        languages_trained = []
        
        for tier_idx, (tier_name, tier_config) in enumerate(self.language_tiers.items()):
            print(f"\n{'='*50}")
            print(f"STAGE {tier_idx + 1}: Training {tier_name}")
            print(f"Languages: {tier_config['languages']}")
            print(f"Reason: {tier_config['reason']}")
            print(f"{'='*50}")
            
            # Add current tier languages
            languages_trained.extend(tier_config['languages'])
            
            # Create dataloader with all languages up to this tier
            train_loader = self.create_tier_dataloader(tier_name)
            
            # Setup optimizer with tier-specific LR
            optimizer = torch.optim.AdamW(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                lr=tier_config['lr'],
                weight_decay=0.01
            )
            
            # Learning rate scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=len(train_loader) * tier_config['epochs']
            )
            
            # Train this tier
            tier_losses = []
            # Resource monitoring for the whole tier
            with resource_monitor.monitor(f"progressive_tier_{tier_name}"):
                for epoch in range(tier_config['epochs']):
                    # ADD shutdown check
                    if shutdown_handler and shutdown_handler.should_stop():
                        logger.info(f"Shutdown requested during tier {tier_name}. Stopping.")
                        self._save_checkpoint(save_dir, tier_name, epoch, languages_trained)
                        return

                    epoch_loss = self._train_epoch(
                        train_loader, 
                        optimizer, 
                        scheduler,
                        epoch,
                        tier_name,
                        use_adapters
                    )
                    tier_losses.append(epoch_loss)
                
                    # Save checkpoint
                    if (epoch + 1) % 2 == 0:
                        self._save_checkpoint(
                            save_dir,
                            tier_name,
                            epoch,
                            languages_trained
                        )
            
                # Validate on all languages seen so far
                val_accuracy = self._validate_tier(languages_trained)
            
                # Store results
                self.training_history['tier_losses'][tier_name] = tier_losses
                self.training_history['tier_accuracies'][tier_name] = val_accuracy
            
                print(f"\nCompleted {tier_name}: Final loss: {tier_losses[-1]:.4f}, "
                      f"Validation accuracy: {val_accuracy:.2%}")

        # Final summary log
        ummary = resource_monitor.get_summary()
        logger.info(f"Progressive training resource summary: {summary}")         

    def _validate_tier(self, languages_trained: List[str], max_samples: int = 5000) -> float:
        """Validate model on specific languages with proper implementation"""
        self.encoder.eval()
        self.decoder.eval()
    
        total_correct = 0
        total_samples = 0
        total_bleu = 0.0
        device = next(self.encoder.parameters()).device
    
        # Create validation dataloader for these languages and Validate with sampling for efficiency
        val_indices = self.get_tier_indices(languages_trained)

        # Sample if too large
        if len(val_indices) > max_samples:
            val_indices = np.random.choice(val_indices, max_samples, replace=False).tolist()

        # Limit validation set size for speed
        max_val_samples = min(1000, len(val_indices))
        val_subset_indices = val_indices[:max_val_samples] if len(val_indices) > max_val_samples else val_indices
    
        val_subset = Subset(self.val_dataset, val_subset_indices)
    
        val_loader = DataLoader(
            val_subset,
            batch_size=32,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        # Import BLEU scorer if available
        try:
            from sacrebleu import corpus_bleu
            use_bleu = True
        except ImportError:
            logger.warning("sacrebleu not available, using token accuracy only")
            use_bleu = False
    
        predictions = []
        references = []
    
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                source_ids = batch['source_ids'].to(device)
                target_ids = batch['target_ids'].to(device)
                source_mask = batch['source_mask'].to(device)

                # Get vocabulary pack
                vocab_pack = batch['vocab_pack']
            
                # Forward pass
                encoder_output = self.encoder(source_ids, source_mask)
            
                # Generate translations with beam search
                generated_ids = self._generate_translations(
                    encoder_output,
                    source_mask,
                    vocab_pack,
                    max_length=target_ids.size(1),
                    beam_size=3
                )

                # Calculate token-level accuracy
                for i in range(generated_ids.size(0)):
                    gen = generated_ids[i]
                    ref = target_ids[i]
                
                    # Find actual lengths (excluding padding)
                    gen_len = (gen != vocab_pack.special_tokens.get('<pad>', 0)).sum().item()
                    ref_len = (ref != vocab_pack.special_tokens.get('<pad>', 0)).sum().item()
                
                    # Exact match accuracy
                    if gen_len == ref_len and torch.equal(gen[:gen_len], ref[:ref_len]):
                        total_correct += 1
                
                    total_samples += 1
                
                    # Store for BLEU calculation
                    if use_bleu:
                        # Detokenize for BLEU
                        gen_text = self._detokenize(gen.cpu().numpy(), vocab_pack)
                        ref_text = self._detokenize(ref.cpu().numpy(), vocab_pack)
                        predictions.append(gen_text)
                        references.append(ref_text)
    
        # Calculate metrics
        accuracy = total_correct / total_samples if total_samples > 0 else 0
    
        if use_bleu and predictions:
            bleu_score = corpus_bleu(predictions, [references]).score
            logger.info(f"Validation - Accuracy: {accuracy:.2%}, BLEU: {bleu_score:.2f}")
            # Combine metrics (weighted average)
            combined_score = 0.3 * accuracy + 0.7 * (bleu_score / 100.0)
        else:
            logger.info(f"Validation - Accuracy: {accuracy:.2%}")
            combined_score = accuracy
    
        return combined_score

    def _generate_translations(self, encoder_output, source_mask, vocab_pack, 
                         max_length=128, beam_size=3):
        """Generate translations using beam search"""
        batch_size = encoder_output.size(0)
        device = encoder_output.device
    
        # Initialize with start token
        start_token = vocab_pack.special_tokens.get('<s>', 2)
        end_token = vocab_pack.special_tokens.get('</s>', 3)
        pad_token = vocab_pack.special_tokens.get('<pad>', 0)
    
        # Simple greedy decoding (replace with beam search for better quality)
        generated = torch.full((batch_size, max_length), pad_token, dtype=torch.long, device=device)
        generated[:, 0] = start_token
    
        for t in range(1, max_length):
            # Decoder forward pass
            decoder_output = self.decoder(
                generated[:, :t],
                encoder_output,
                encoder_attention_mask=source_mask
            )
        
            # Get next token (greedy)
            next_token = decoder_output[:, -1, :].argmax(dim=-1)
            generated[:, t] = next_token
        
            # Check if all sequences have ended
            if (next_token == end_token).all():
                break
    
        return generated

    def _detokenize(self, token_ids, vocab_pack):
        """Convert token IDs to text"""
        # Create reverse mapping
        id_to_token = {v: k for k, v in vocab_pack.tokens.items()}
        id_to_token.update({v: k for k, v in vocab_pack.special_tokens.items()})
    
        tokens = []
        for token_id in token_ids:
            if token_id == vocab_pack.special_tokens.get('</s>', 3):
                break
            if token_id == vocab_pack.special_tokens.get('<pad>', 0):
                continue
        
            token = id_to_token.get(int(token_id), '<unk>')
            if not token.startswith('<'):  # Skip special tokens
                tokens.append(token)
    
        return ' '.join(tokens) 
    
    def _train_epoch(self, dataloader, optimizer, scheduler, epoch, tier_name, use_adapters):
        """Train one epoch"""
        self.encoder.train()
        self.decoder.train()
        
        total_loss = 0
        device = next(self.encoder.parameters()).device
        
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            source_ids = batch['source_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            source_mask = batch['source_mask'].to(device)
            
            # Get source language for adapter
            source_lang = batch['metadata']['source_lang'][0]  # Assuming same in batch
            
            # Load adapter if using
            if use_adapters and hasattr(self.encoder, 'load_language_adapter'):
                self.encoder.load_language_adapter(source_lang)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Encode
            if use_adapters and hasattr(self.encoder, 'language_adapters'):
                encoder_output = self.encoder(source_ids, source_mask, source_lang)
            else:
                encoder_output = self.encoder(source_ids, source_mask)
            
            # Decode
            decoder_output = self.decoder(
                target_ids[:, :-1],
                encoder_output,
                encoder_attention_mask=source_mask
            )
            
            # Loss
            loss = torch.nn.functional.cross_entropy(
                decoder_output.reshape(-1, decoder_output.size(-1)),
                target_ids[:, 1:].reshape(-1),
                ignore_index=0
            )
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                max_norm=1.0
            )
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            # Log
            if batch_idx % 100 == 0:
                print(f"Tier: {tier_name}, Epoch: {epoch+1}, "
                      f"Batch: {batch_idx}/{len(dataloader)}, "
                      f"Loss: {loss.item():.4f}, "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
        return total_loss / len(dataloader)

    def _train_step(self, batch: Dict[str, Any], optimizer: torch.optim.Optimizer) -> float:
        """Single training step"""
        device = next(self.encoder.parameters()).device
    
        # Move batch to device
        source_ids = batch['source_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        source_mask = batch['source_mask'].to(device)
    
        # Forward pass
        encoder_output = self.encoder(source_ids, source_mask)
        decoder_output = self.decoder(
            target_ids[:, :-1],
            encoder_output,
            encoder_attention_mask=source_mask
        )
    
        # Calculate loss
        loss = torch.nn.functional.cross_entropy(
            decoder_output.reshape(-1, decoder_output.size(-1)),
            target_ids[:, 1:].reshape(-1),
            ignore_index=0
        )
    
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            max_norm=1.0
        )
        optimizer.step()
    
        return loss.item()   
    
    def _save_checkpoint(self, save_dir, tier_name, epoch, languages_trained):
        """Save progressive checkpoint"""
        checkpoint = {
            'tier': tier_name,
            'epoch': epoch,
            'languages_trained': languages_trained,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'training_history': self.training_history
        }
        
        save_path = Path(save_dir) / f'{tier_name}_epoch_{epoch}.pt'
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint: {save_path}")

# Add to ProgressiveTrainingStrategy
class TrainingState:
    """Persistent training state across interruptions"""
    
    def __init__(self, state_file: str = "training_state.json"):
        self.state_file = state_file
        self.state = self.load_state()
    
    def load_state(self) -> Dict:
        """Load training state from disk"""
        if Path(self.state_file).exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            'current_tier': 0,
            'current_epoch': 0,
            'completed_tiers': [],
            'total_steps': 0
        }
    
    def save_state(self):
        """Save training state to disk"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def update(self, **kwargs):
        """Update state and save"""
        self.state.update(kwargs)
        self.save_state()

    # Add to progressive_training.py
    def find_learning_rate(self, tier_name: str, num_iterations: int = 100) -> float:
        """Find optimal learning rate for a tier"""
        import matplotlib.pyplot as plt
    
        dataloader = self.create_tier_dataloader(tier_name)
    
        # Setup
        min_lr = 1e-7
        max_lr = 1e-1
        lrs = np.logspace(np.log10(min_lr), np.log10(max_lr), num_iterations)
        losses = []
    
        # Temporary optimizer
        optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=min_lr
        )
    
        # Learning rate finder loop
        for i, (batch, lr) in enumerate(zip(dataloader, lrs)):
            if i >= num_iterations:
                break
            
            # Update LR
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
            # Train step
            loss = self._train_step(batch, optimizer)
            losses.append(loss)
        
            # Stop if loss explodes
            if loss > min(losses) * 4:
                break
    
        # Find best LR (steepest descent)
        gradients = np.gradient(losses)
        best_idx = np.argmin(gradients)
        best_lr = lrs[best_idx] / 10  # Use 1/10th of steepest point
    
        logger.info(f"Best learning rate for {tier_name}: {best_lr:.2e}")
        return best_lr            

# Update train_universal_system.py to use progressive training
def train_with_progressive_strategy():
    """Main training function using progressive strategy"""
    
    # Load models
    from encoder.universal_encoder import UniversalEncoder
    from cloud_decoder.optimized_decoder import OptimizedUniversalDecoder
    
    encoder = UniversalEncoder()
    decoder = OptimizedUniversalDecoder()
    
    # Load datasets
    train_dataset = ModernParallelDataset('data/processed/train_final.txt')
    val_dataset = ModernParallelDataset('data/processed/val_final.txt')

    def cleanup():
        """Emergency cleanup for progressive training"""
        if 'progressive_trainer' in locals():
            logger.info("Saving emergency progressive checkpoint...")
            progressive_trainer._save_checkpoint(
                save_dir='checkpoints/progressive',
                tier_name='emergency',
                epoch=0,
                languages_trained=[]
            )
            
    shutdown_handler = GracefulShutdown(cleanup_func=cleanup)
    versioning = ModelVersion()    
    
    # Create progressive trainer
    progressive_trainer = ProgressiveTrainingStrategy(
        encoder=encoder,
        decoder=decoder,
        train_dataset=train_dataset,
        val_dataset=val_dataset
    )
    
    # Start progressive training
    progressive_trainer.train_progressive(
        save_dir='checkpoints/progressive',
        use_adapters=True,  # Enable adapter training
        shutdown_handler=shutdown_handler
    )

    # Register model versioning after training
    final_encoder_path = "checkpoints/progressive/final_encoder.pt" # Assuming you save a final model
    if Path(final_encoder_path).exists():
        version = versioning.register_model(
            model_path=final_encoder_path,
            model_type="encoder-progressive",
            metrics={'final_accuracy': progressive_trainer.training_history['tier_accuracies']['tier4']}
        )
        logger.info(f"Progressive model registered as version: {version}")