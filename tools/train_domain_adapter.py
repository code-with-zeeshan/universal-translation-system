# tools/train_domain_adapter.py

import logging
from torch.utils.data import DataLoader
from encoder.train_adapters import AdapterTrainer
from data.dataset_classes import ModernParallelDataset # Assuming this is your dataset class
from vocabulary.vocabulary_manager import VocabularyManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_for_domain(
    domain: str,
    language: str,
    base_model_path: str = "models/production/encoder.pt",
    epochs: int = 5
):
    """
    Orchestrates the training of a domain-specific adapter.
    
    Example: train_for_domain('medical', 'es')
    """
    logger.info(f"🚀 Starting domain adapter training for language '{language}' in domain '{domain}'")

    # 1. Initialize the Adapter Trainer with the general-purpose base model
    trainer = AdapterTrainer(base_model_path=base_model_path)

    # 2. Load the domain-specific vocabulary
    # The manager will load 'latin_medical_v1.0.msgpack' (or latest version)
    vocab_manager = VocabularyManager()
    domain_pack_name = f"latin_{domain}" # Construct the pack name
    
    try:
        # This assumes you have a way to load a specific pack by name
        # We might need a small addition to VocabularyManager if not
        domain_vocab = vocab_manager._load_pack(domain_pack_name)
        logger.info(f"Loaded domain vocabulary: {domain_pack_name} v{domain_vocab.version}")
    except Exception as e:
        logger.error(f"Failed to load domain vocabulary '{domain_pack_name}'. Did you create it first? Error: {e}")
        return

    # 3. Create a DataLoader using only the domain-specific data
    # This points to the parallel data file for the domain
    domain_data_path = f"data/processed/{language}_{domain}_parallel.txt"
    
    domain_dataset = ModernParallelDataset(
        data_path=domain_data_path,
        # Pass the domain vocab to the dataset so it tokenizes correctly
        vocab_manager=None, # Or adapt to pass the specific pack
    )
    # You might need to adapt your dataset class to accept a pre-loaded vocab pack
    
    train_loader = DataLoader(domain_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(domain_dataset, batch_size=32) # Use a separate validation set in production

    # 4. Train the new adapter
    adapter_name = f"{language}_{domain}"
    result = trainer.train_adapter(
        language=adapter_name, # Use a unique name like 'es_medical'
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs
    )

    logger.info(f"✅ Training complete for adapter '{adapter_name}'!")
    logger.info(f"  Best validation loss: {result['best_val_loss']:.4f}")
    logger.info(f"  Adapter saved to: {result['adapter_path']}")

if __name__ == "__main__":
    # Example: Train a Spanish medical adapter
    # Pre-requisites:
    # 1. Run data pipeline on medical data to create 'data/processed/es_medical_parallel.txt'
    # 2. Run 'tools/create_vocabulary_packs.py' to create 'latin_medical' pack
    train_for_domain(domain="medical", language="es")