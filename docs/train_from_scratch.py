# docs/train_from_scratch.py
"""
Modern training script for encoder/decoder from scratch, aligned with the Universal Translation System pipeline.
- Uses environment variable configuration
- Supports config auto-detection for hardware
- Integrates with model conversion and deployment best practices
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import argparse
import os
from config.schemas import load_config

# 1. Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train Universal Translation System from scratch")
    parser.add_argument("--config", type=str, default=None, help="Path to config file (optional, auto-detected if not provided)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (overrides config)")
    parser.add_argument("--output_dir", type=str, default="models/production", help="Directory to save models")
    parser.add_argument("--log_wandb", action="store_true", help="Log metrics to Weights & Biases")
    return parser.parse_args()

# 2. Auto-select best config for hardware
def auto_select_config():
    if not torch.cuda.is_available():
        return "config/training_cpu.yaml"
    
    gpu_name = torch.cuda.get_device_name(0)
    if "A100" in gpu_name:
        return "config/training_a100.yaml"
    if "V100" in gpu_name:
        return "config/training_v100.yaml"
    if "3090" in gpu_name:
        return "config/training_rtx3090.yaml"
    if "T4" in gpu_name:
        return "config/training_t4.yaml"
    
    # Default to T4 config for unknown GPUs
    return "config/training_t4.yaml"

# 3. Define the trainer class
class UniversalModelTrainer:
    def __init__(self, config):
        from encoder.universal_encoder import UniversalEncoder
        from cloud_decoder.optimized_decoder import OptimizedUniversalDecoder
        
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.encoder = UniversalEncoder(**config['training'])
        self.decoder = OptimizedUniversalDecoder(**config['training'])
        
        # Move models to device
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        
        # Use DataParallel for multi-GPU training
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training")
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)
        
        # Initialize optimizers
        self.encoder_optimizer = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=config['training'].get('learning_rate', 5e-5),
            weight_decay=config['training'].get('weight_decay', 0.01)
        )
        
        self.decoder_optimizer = torch.optim.AdamW(
            self.decoder.parameters(),
            lr=config['training'].get('learning_rate', 5e-5),
            weight_decay=config['training'].get('weight_decay', 0.01)
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def train(self, train_dataloader, val_dataloader, num_epochs=20, output_dir="models/production"):
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup learning rate scheduler
        total_steps = len(train_dataloader) * num_epochs
        encoder_scheduler = get_linear_schedule_with_warmup(
            self.encoder_optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )
        
        decoder_scheduler = get_linear_schedule_with_warmup(
            self.decoder_optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )
        
        # Training loop
        for epoch in range(num_epochs):
            self.encoder.train()
            self.decoder.train()
            
            total_loss = 0
            for batch in train_dataloader:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)
                
                # Zero gradients
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
                
                # Forward pass
                encoder_outputs = self.encoder(input_ids, attention_mask)
                decoder_outputs = self.decoder(
                    encoder_outputs=encoder_outputs,
                    encoder_attention_mask=attention_mask,
                    decoder_input_ids=target_ids[:, :-1]  # Shift right for teacher forcing
                )
                
                # Calculate loss
                loss = self.criterion(
                    decoder_outputs.view(-1, decoder_outputs.size(-1)),
                    target_ids[:, 1:].contiguous().view(-1)  # Shift left for targets
                )
                
                # Backward pass
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
                
                # Update weights
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()
                
                # Update learning rate
                encoder_scheduler.step()
                decoder_scheduler.step()
                
                total_loss += loss.item()
            
            # Calculate average loss for the epoch
            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_loss:.4f}")
            
            # Validation
            val_loss = self.validate(val_dataloader)
            print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                self.save_checkpoint(output_dir, f"checkpoint_epoch_{epoch+1}")
        
        # Save final models
        self.save_models(output_dir)
        
        return {
            "final_train_loss": avg_loss,
            "final_val_loss": val_loss
        }
    
    def validate(self, val_dataloader):
        self.encoder.eval()
        self.decoder.eval()
        
        total_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)
                
                # Forward pass
                encoder_outputs = self.encoder(input_ids, attention_mask)
                decoder_outputs = self.decoder(
                    encoder_outputs=encoder_outputs,
                    encoder_attention_mask=attention_mask,
                    decoder_input_ids=target_ids[:, :-1]  # Shift right for teacher forcing
                )
                
                # Calculate loss
                loss = self.criterion(
                    decoder_outputs.view(-1, decoder_outputs.size(-1)),
                    target_ids[:, 1:].contiguous().view(-1)  # Shift left for targets
                )
                
                total_loss += loss.item()
        
        return total_loss / len(val_dataloader)
    
    def save_checkpoint(self, output_dir, checkpoint_name):
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save encoder
        encoder_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_encoder.pt")
        torch.save({
            'model_state_dict': self.encoder.state_dict() if not isinstance(self.encoder, nn.DataParallel) else self.encoder.module.state_dict(),
            'optimizer_state_dict': self.encoder_optimizer.state_dict()
        }, encoder_path)
        
        # Save decoder
        decoder_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_decoder.pt")
        torch.save({
            'model_state_dict': self.decoder.state_dict() if not isinstance(self.decoder, nn.DataParallel) else self.decoder.module.state_dict(),
            'optimizer_state_dict': self.decoder_optimizer.state_dict()
        }, decoder_path)
        
        print(f"Saved checkpoint to {checkpoint_dir}")
    
    def save_models(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        # Save encoder
        encoder_path = os.path.join(output_dir, "encoder.pt")
        torch.save(
            self.encoder.module if isinstance(self.encoder, nn.DataParallel) else self.encoder,
            encoder_path
        )
        
        # Save decoder
        decoder_path = os.path.join(output_dir, "decoder.pt")
        torch.save(
            self.decoder.module if isinstance(self.decoder, nn.DataParallel) else self.decoder,
            decoder_path
        )
        
        print(f"Saved final models to {output_dir}")

# 4. Model conversion utilities
class ModelConverter:
    @staticmethod
    def pytorch_to_onnx(model_path, output_path, dynamic_axes=True):
        model = torch.load(model_path)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randint(0, 100, (1, 10))
        
        # Define dynamic axes if needed
        dynamic_axes_dict = None
        if dynamic_axes:
            dynamic_axes_dict = {
                'input_ids': {0: 'batch', 1: 'sequence'},
                'encoder_output': {0: 'batch', 1: 'sequence'}
            }
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input_ids'],
            output_names=['encoder_output'],
            dynamic_axes=dynamic_axes_dict,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
        )
        
        # Optimize ONNX model
        try:
            import onnx
            from onnxsim import simplify
            
            model = onnx.load(output_path)
            model_sim, check = simplify(model)
            assert check, "Simplified ONNX model could not be validated"
            onnx.save(model_sim, output_path)
            print(f"Exported and optimized ONNX model to {output_path}")
        except ImportError:
            print("ONNX simplification skipped: onnx or onnxsim not installed")
    
    @staticmethod
    def onnx_to_coreml(onnx_path, output_path):
        try:
            import coremltools as ct
            
            model = ct.convert(
                onnx_path,
                minimum_ios_deployment_target='14',
                compute_precision=ct.precision.FLOAT16,
                convert_to="neuralnetwork"
            )
            model.save(output_path)
            print(f"Converted ONNX to Core ML: {output_path}")
        except ImportError:
            print("Core ML conversion skipped: coremltools not installed")
    
    @staticmethod
    def onnx_to_tflite(onnx_path, output_path):
        try:
            import tensorflow as tf
            import onnx
            from onnx_tf.backend import prepare
            
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            
            # Convert to TensorFlow
            tf_rep = prepare(onnx_model)
            tf_rep.export_graph("temp_tf_model")
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model("temp_tf_model")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            
            tflite_model = converter.convert()
            
            # Save TFLite model
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            # Clean up temporary files
            import shutil
            if os.path.exists("temp_tf_model"):
                shutil.rmtree("temp_tf_model")
            
            print(f"Converted ONNX to TFLite: {output_path}")
        except ImportError:
            print("TFLite conversion skipped: tensorflow, onnx, or onnx-tf not installed")

# 5. Data loading helper
def load_datasets(config):
    from torch.utils.data import DataLoader
    
    # Import dataset class
    from utils.dataset_classes import UniversalTranslationDataset
    
    # Load vocabulary
    from vocabulary.unified_vocab_manager import UnifiedVocabularyManager, VocabularyMode
    vocab_manager = UnifiedVocabularyManager(config, mode=VocabularyMode.FULL)
    
    # Create datasets
    train_dataset = UniversalTranslationDataset(
        data_dir=config['data']['processed_dir'],
        split="train",
        vocab_manager=vocab_manager,
        max_length=config['training'].get('max_length', 128)
    )
    
    val_dataset = UniversalTranslationDataset(
        data_dir=config['data']['processed_dir'],
        split="val",
        vocab_manager=vocab_manager,
        max_length=config['training'].get('max_length', 128)
    )
    
    # Create dataloaders
    batch_size = config['training'].get('batch_size', 32)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_dataloader, val_dataloader

# 6. Main entry point
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Auto-select config if not provided
    config_path = args.config if args.config else auto_select_config()
    print(f"Using config: {config_path}")
    
    # Load config
    config = load_config(config_path)
    config_dict = config.dict()
    
    # Override batch size if provided
    if args.batch_size:
        config_dict['training']['batch_size'] = args.batch_size
    
    # Initialize Weights & Biases if requested
    if args.log_wandb:
        try:
            import wandb
            wandb.init(
                project="universal-translation-system",
                config=config_dict
            )
        except ImportError:
            print("wandb not installed. Skipping logging.")
    
    # Load datasets
    print("Loading datasets...")
    train_dataloader, val_dataloader = load_datasets(config_dict)
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = UniversalModelTrainer(config_dict)
    
    # Train model
    print(f"Starting training for {args.epochs} epochs...")
    results = trainer.train(
        train_dataloader,
        val_dataloader,
        num_epochs=args.epochs,
        output_dir=args.output_dir
    )
    
    # Print final results
    print("\nTraining completed!")
    print(f"Final training loss: {results['final_train_loss']:.4f}")
    print(f"Final validation loss: {results['final_val_loss']:.4f}")
    
    # Convert models to ONNX
    print("\nConverting models to ONNX...")
    encoder_path = os.path.join(args.output_dir, "encoder.pt")
    decoder_path = os.path.join(args.output_dir, "decoder.pt")
    
    encoder_onnx_path = os.path.join(args.output_dir, "encoder.onnx")
    decoder_onnx_path = os.path.join(args.output_dir, "decoder.onnx")
    
    ModelConverter.pytorch_to_onnx(encoder_path, encoder_onnx_path)
    ModelConverter.pytorch_to_onnx(decoder_path, decoder_onnx_path)
    
    print("\nTraining and conversion complete!")
    print(f"Models saved to: {args.output_dir}")