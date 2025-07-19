# training/progressive_training.py
import torch
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Optional
import numpy as np
from pathlib import Path

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
                         use_adapters: bool = True):
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
            for epoch in range(tier_config['epochs']):
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

    def _validate_tier(self, languages_trained: List[str]) -> float:
        """Validate model on specific languages"""
        self.encoder.eval()
        self.decoder.eval()
    
        total_correct = 0
        total_samples = 0
        device = next(self.encoder.parameters()).device
    
        # Create validation dataloader for these languages
        val_indices = self.get_tier_indices(languages_trained)
        val_subset = Subset(self.val_dataset, val_indices[:1000])  # Sample 1000 for speed
    
        val_loader = DataLoader(
            val_subset,
            batch_size=32,
            shuffle=False,
            num_workers=2
        )
    
        with torch.no_grad():
            for batch in val_loader:
                source_ids = batch['source_ids'].to(device)
                target_ids = batch['target_ids'].to(device)
                source_mask = batch['source_mask'].to(device)
            
                # Forward pass
                encoder_output = self.encoder(source_ids, source_mask)
            
                # Generate translations (greedy decoding for speed)
                generated = self.decoder.generate(
                    encoder_output,
                    source_mask,
                    max_length=target_ids.size(1)
                )
            
                # Calculate accuracy (simplified - exact match)
                matches = (generated == target_ids).all(dim=1).sum().item()
                total_correct += matches
                total_samples += source_ids.size(0)
    
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        return accuracy     
    
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
        use_adapters=True  # Enable adapter training
    )