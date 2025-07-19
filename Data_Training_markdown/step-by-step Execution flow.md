# Universal Translation Training - Execution Flow

## Step 1: Initialize Models (REQUIRED FIRST)
```bash
python bootstrap_from_pretrained.py
```
**Creates:**
- `models/universal_encoder_initial.pt`
- `models/universal_decoder_initial.pt`
- Vocabulary mappings in memory

**Dependencies:** None (starting point)

---

## Step 2: Prepare Training Data
```bash
# You need to create/prepare your parallel training data
# This is referenced but not implemented in the scripts
python prepare_training_data.py  # Not shown in files
```

---

## Step 3: Configure Training Environment

### Option A: Single GPU Training
```bash
# Configure memory optimizations
python -c "
from memory_efficient_training import optimize_memory_usage
optimize_memory_usage()
"
```

### Option B: Multi-GPU Training
```bash
# Set up distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=4

torchrun --nproc_per_node=4 distributed_train.py
```

---

## Step 4: Main Training (CORE SCRIPT)
```bash
python train_universal_system.py
```

**This script:**
- Loads models from Step 1
- Applies configurations from Step 3
- Runs complete training loop
- Saves checkpoints periodically

**Creates:**
- `checkpoints/checkpoint_epoch_1.pt`
- `checkpoints/checkpoint_epoch_2.pt`
- `checkpoints/best_model.pt`
- Wandb training logs

---

## Step 5: Model Conversion (DEPLOYMENT)
```bash
python convert_models.py
```

**Converts:**
- `checkpoints/best_model.pt` → `model.onnx`
- `model.onnx` → `model.mlmodel` (iOS)
- `model.onnx` → `model.tflite` (Android)

---

## Complete Workflow Command Sequence

```bash
# 1. Bootstrap models from pretrained weights
python bootstrap_from_pretrained.py

# 2. Train the universal system
python train_universal_system.py

# 3. Convert trained models for deployment
python convert_models.py
```

## File Dependencies

### `bootstrap_from_pretrained.py`
- **Imports:** `transformers`, `torch`
- **Creates:** Model initialization files
- **Dependencies:** None (starting point)

### `train_universal_system.py`
- **Imports:** Models from bootstrap phase
- **Uses:** `distributed_train.py`, `memory_efficient_training.py`
- **Creates:** Training checkpoints
- **Dependencies:** Must run after bootstrap

### `convert_models.py`
- **Imports:** Trained model checkpoints
- **Creates:** Deployment-ready model files
- **Dependencies:** Must run after training

### `distributed_train.py` & `memory_efficient_training.py`
- **Role:** Configuration modules
- **Used by:** `train_universal_system.py`
- **Dependencies:** Can be developed in parallel

## Key Integration Points

1. **Model State Transfer:**
   ```python
   # bootstrap → training
   encoder = torch.load('models/universal_encoder_initial.pt')
   
   # training → conversion
   model = torch.load('checkpoints/best_model.pt')
   ```

2. **Vocabulary Management:**
   ```python
   # Shared across all phases
   vocab_pack = VocabularyManager().get_vocab_for_pair(src_lang, tgt_lang)
   ```

3. **Configuration Injection:**
   ```python
   # Infrastructure scripts configure main trainer
   trainer.setup_distributed()  # from distributed_train.py
   trainer.optimize_memory()    # from memory_efficient_training.py
   ```