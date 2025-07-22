# encoder/train_adapters.py
"""Train language-specific adapters efficiently"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from encoder.language_adapters import AdapterUniversalEncoder

logger = logging.getLogger(__name__)

class AdapterTrainer:
    """Train only language adapters while keeping base model frozen"""
    
    def __init__(self, base_model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = AdapterUniversalEncoder(base_model_path=base_model_path)
        self.model.to(self.device)
        
        # Ensure base model is frozen
        self.model.freeze_base_encoder()
        
        logger.info(f"Initialized AdapterTrainer on {self.device}")
        logger.info(f"Base model parameters: {sum(p.numel() for p in self.model.base_encoder.parameters()):,}")
        
    def train_adapter(self, 
                     language: str, 
                     train_loader: DataLoader, 
                     val_loader: DataLoader, 
                     epochs: int = 5,
                     lr: float = 1e-4,
                     warmup_steps: int = 500) -> Dict[str, Any]:
        """Train adapter for specific language"""
        
        # Add adapter for this language
        self.model.add_language_adapter(language)
        
        # Get adapter parameters only
        adapter_params = list(self.model.language_adapters[language].parameters())
        
        # Verify only adapter params require grad
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        adapter_param_count = sum(p.numel() for p in adapter_params)
        logger.info(f"Training {language} adapter: {adapter_param_count:,} parameters")
        logger.info(f"Total trainable parameters: {trainable_params:,}")
        
        # Optimizer for adapter only
        optimizer = torch.optim.AdamW(adapter_params, lr=lr, weight_decay=0.01)
        
        # Learning rate scheduler
        total_steps = len(train_loader) * epochs
        scheduler = self._get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            train_loss = self._train_epoch(
                train_loader, optimizer, scheduler, language, epoch, epochs
            )
            history['train_loss'].append(train_loss)
            
            # Validation
            val_loss, val_accuracy = self._validate(val_loader, language)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            logger.info(f"Epoch {epoch+1}/{epochs}")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}")
            logger.info(f"  Val Accuracy: {val_accuracy:.2%}")
            
            # Save best adapter
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = f'models/adapters/best_{language}_adapter.pt'
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                self.model.save_language_adapter(language, save_path)
                logger.info(f"  💾 Saved best adapter to {save_path}")
        
        # Save final adapter
        final_path = f'models/adapters/final_{language}_adapter.pt'
        self.model.save_language_adapter(language, final_path)
        
        return {
            'best_val_loss': best_val_loss,
            'history': history,
            'language': language,
            'adapter_path': final_path
        }
    
    def _train_epoch(self, train_loader, optimizer, scheduler, language, epoch, total_epochs):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch.get('labels', input_ids).to(self.device)  # For MLM task
            
            # Forward pass with adapter
            outputs = self.model(input_ids, attention_mask, language)
            
            # Compute loss
            loss = self._compute_loss(outputs, labels, attention_mask)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.language_adapters[language].parameters(), 
                max_norm=1.0
            )
            
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            current_lr = scheduler.get_last_lr()[0]
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.2e}'
            })
        
        return total_loss / len(train_loader)
    
    @torch.no_grad()
    def _validate(self, val_loader, language):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch.get('labels', input_ids).to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask, language)
            
            # Compute loss
            loss = self._compute_loss(outputs, labels, attention_mask)
            total_loss += loss.item()
            
            # Compute accuracy (simplified - adapt based on your task)
            predictions = outputs.argmax(dim=-1)
            mask = attention_mask.bool()
            correct = (predictions == labels) & mask
            total_correct += correct.sum().item()
            total_samples += mask.sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        return avg_loss, accuracy
    
    def _compute_loss(self, outputs, labels, attention_mask):
        """Compute task-specific loss (production)."""
        # For MLM or sequence-to-sequence, use CrossEntropyLoss
        vocab_size = outputs.size(-1)
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        logits = outputs.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        masked_lm_loss = loss_fct(logits, labels_flat)
        return masked_lm_loss
    
    def _get_linear_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps):
        """Create a schedule with a learning rate that decreases linearly after warmup"""
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def train_all_adapters(self, languages: list, train_loaders: dict, val_loaders: dict, 
                          epochs: int = 5) -> Dict[str, Any]:
        """Train adapters for multiple languages"""
        results = {}
        
        for lang in languages:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training adapter for {lang}")
            logger.info(f"{'='*50}")
            
            if lang not in train_loaders or lang not in val_loaders:
                logger.warning(f"Skipping {lang} - missing data loaders")
                continue
            
            result = self.train_adapter(
                language=lang,
                train_loader=train_loaders[lang],
                val_loader=val_loaders[lang],
                epochs=epochs
            )
            
            results[lang] = result
        
        return results


# Usage example
if __name__ == "__main__":
    # Initialize trainer
    trainer = AdapterTrainer(base_model_path="models/universal_encoder.pt")
    
    # Create dummy data loader for testing
    from torch.utils.data import TensorDataset
    
    # Dummy data
    input_ids = torch.randint(0, 1000, (100, 128))
    attention_mask = torch.ones(100, 128)
    labels = torch.randint(0, 1000, (100, 128))
    
    dataset = TensorDataset(input_ids, attention_mask, labels)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Train adapter for Spanish
    result = trainer.train_adapter(
        language='es',
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3
    )
    
    print(f"Training complete! Best validation loss: {result['best_val_loss']:.4f}")