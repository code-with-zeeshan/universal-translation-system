# training/train_universal_system.py
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
import wandb
from tqdm import tqdm

class UniversalSystemTrainer:
    def __init__(self, encoder, decoder, train_data_path, val_data_path):
        self.encoder = encoder
        self.decoder = decoder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move models to device
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        
        # Create datasets
        self.train_dataset = ParallelDataset(train_data_path)
        self.val_dataset = ParallelDataset(val_data_path)
        
        # Optimizers
        self.encoder_optimizer = torch.optim.AdamW(
            self.encoder.parameters(), lr=5e-5, weight_decay=0.01
        )
        self.decoder_optimizer = torch.optim.AdamW(
            self.decoder.parameters(), lr=5e-5, weight_decay=0.01
        )
        
        # Initialize wandb for tracking
        wandb.init(project="universal-translation", name="training-run-1")
    
    def train(self, num_epochs=10):
        """Complete training loop"""
        
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=32,
            shuffle=True,
            num_workers=4
        )
        
        # Learning rate schedulers
        num_training_steps = len(train_loader) * num_epochs
        encoder_scheduler = get_cosine_schedule_with_warmup(
            self.encoder_optimizer,
            num_warmup_steps=1000,
            num_training_steps=num_training_steps
        )
        decoder_scheduler = get_cosine_schedule_with_warmup(
            self.decoder_optimizer,
            num_warmup_steps=1000,
            num_training_steps=num_training_steps
        )
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"\n🏃 Epoch {epoch+1}/{num_epochs}")
            
            self.encoder.train()
            self.decoder.train()
            
            epoch_loss = 0
            for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
                # Move batch to device
                source_ids = batch['source_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                source_mask = batch['source_mask'].to(self.device)
                target_mask = batch['target_mask'].to(self.device)
                
                # Dynamic vocabulary loading for this batch
                vocab_pack = batch['vocab_pack']
                self.encoder.load_vocabulary_pack(vocab_pack)
                self.decoder.load_vocabulary_pack(vocab_pack)
                
                # Forward pass
                encoder_output = self.encoder(source_ids, source_mask)
                
                # Decoder forward (teacher forcing)
                decoder_output = self.decoder(
                    target_ids[:, :-1],
                    encoder_output,
                    encoder_attention_mask=source_mask
                )
                
                # Calculate loss
                loss = nn.functional.cross_entropy(
                    decoder_output.reshape(-1, decoder_output.size(-1)),
                    target_ids[:, 1:].reshape(-1),
                    ignore_index=vocab_pack.tokens['<pad>']
                )
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.0)
                
                # Optimizer steps
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()
                encoder_scheduler.step()
                decoder_scheduler.step()
                
                # Zero gradients
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
                
                # Logging
                epoch_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/encoder_lr': encoder_scheduler.get_last_lr()[0],
                        'train/decoder_lr': decoder_scheduler.get_last_lr()[0],
                    })
            
            # Validation
            val_loss = self.validate()
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss)
            
            print(f"Epoch {epoch+1} - Train Loss: {epoch_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
    
    def validate(self):
        """Validation loop"""
        self.encoder.eval()
        self.decoder.eval()
        
        val_loader = DataLoader(self.val_dataset, batch_size=64)
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Similar to training but without gradients
                source_ids = batch['source_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                
                encoder_output = self.encoder(source_ids, batch['source_mask'])
                decoder_output = self.decoder(
                    target_ids[:, :-1],
                    encoder_output,
                    encoder_attention_mask=batch['source_mask']
                )
                
                loss = nn.functional.cross_entropy(
                    decoder_output.reshape(-1, decoder_output.size(-1)),
                    target_ids[:, 1:].reshape(-1),
                    ignore_index=0  # padding
                )
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'encoder_optimizer': self.encoder_optimizer.state_dict(),
            'decoder_optimizer': self.decoder_optimizer.state_dict(),
            'val_loss': val_loss,
        }
        
        torch.save(checkpoint, f'checkpoints/checkpoint_epoch_{epoch}.pt')
        
        # Also save best model
        if not hasattr(self, 'best_val_loss') or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(checkpoint, 'checkpoints/best_model.pt')

# Create data processing pipeline
class ParallelDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        # Load preprocessed parallel data
        self.data = self._load_data(data_path)
        self.vocab_manager = VocabularyManager()
        
    def _load_data(self, path):
        """Load parallel sentences"""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 4:
                    data.append({
                        'source': parts[0],
                        'target': parts[1],
                        'source_lang': parts[2],
                        'target_lang': parts[3]
                    })
        return data
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get vocabulary pack for this language pair
        vocab_pack = self.vocab_manager.get_vocab_for_pair(
            item['source_lang'], 
            item['target_lang']
        )
        
        # Tokenize (simplified)
        source_tokens = self._tokenize(item['source'], item['source_lang'], vocab_pack)
        target_tokens = self._tokenize(item['target'], item['target_lang'], vocab_pack)
        
        return {
            'source_ids': torch.tensor(source_tokens),
            'target_ids': torch.tensor(target_tokens),
            'source_mask': torch.tensor([1] * len(source_tokens)),
            'target_mask': torch.tensor([1] * len(target_tokens)),
            'vocab_pack': vocab_pack
        }