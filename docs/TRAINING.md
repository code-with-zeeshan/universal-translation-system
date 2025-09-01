# Training Guide

## Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 50GB+ free disk space
- 16GB+ RAM

## 1. Data Preparation
1. Run the unified data pipeline:
   ```bash
   python -m data.unified_data_pipeline
   ```
   - Processed files are expected under `data/processed/` (e.g., `train_final.txt`, `val_final.txt`, `test_final.txt`).
   - The dataset initializes UnifiedVocabularyManager using your loaded configuration; ensure `vocabulary.vocab_dir` and `vocabulary.language_to_pack_mapping` are correctly set in your config.
2. Create or evolve vocabulary packs if needed:
   ```bash
   # Create a unified vocabulary
   python vocabulary/unified_vocabulary_creator.py

   # Or evolve/update an existing vocabulary
   python vocabulary/evolve_vocabulary.py
   ```
   - Configure vocabulary dir in your config: `vocabulary.vocab_dir`.

## 2. Configuration
- Base and hardware-aware configs live in `config/`.
  - Common options: `config/base.yaml`, `config/training_generic_gpu.yaml`, `config/training_generic_multi_gpu.yaml`.
- Key fields used by the launcher:
  - **data.processed_dir**: path containing `train_final.txt`, `val_final.txt`, `test_final.txt`
  - **data.cache_dir**: optional dataset cache
  - **vocabulary.vocab_dir**: directory with vocabulary packs
  - **model**: `vocab_size`, `hidden_dim`, `num_layers`, `num_heads`, `decoder_dim`, `decoder_layers`, `decoder_heads`, `dropout`
  - **training**: `batch_size`, `learning_rate`, `num_epochs`, `checkpoint_dir`
  - **monitoring**: `use_wandb` etc.

## 3. Training (Intelligent Trainer)
Use the consolidated launcher with subcommands.

- Basic single-GPU/CPU training:
  ```bash
  python -m training.launch train --config config/base.yaml
  ```

- Dynamic config (no YAML):
  ```bash
  python -m training.launch train --config dynamic --dynamic
  ```

- Archived GPU configs (hardware-aware presets):
  ```bash
  # Examples
  python -m training.launch train --config config/archived_gpu_configs/training_generic_gpu.yaml
  python -m training.launch train --config config/archived_gpu_configs/training_t4.yaml
  python -m training.launch train --config config/archived_gpu_configs/training_a100.yaml
  ```

- Override common hyperparameters from CLI:
  ```bash
  python -m training.launch train \
    --config config/training_generic_gpu.yaml \
    --batch-size 64 \
    --learning-rate 6e-4 \
    --num-epochs 10 \
    --experiment-name exp_generic_gpu
  ```

- Resume from a checkpoint:
  ```bash
  python -m training.launch train \
    --config config/base.yaml \
    --checkpoint checkpoints/exp_generic_gpu/best_model.pt
  ```

- Distributed training (DDP) on multiple GPUs:
  ```bash
  python -m training.launch train \
    --config config/training_generic_multi_gpu.yaml \
    --distributed \
    --experiment-name exp_multi
  ```
  - Uses PyTorch distributed with `nccl` backend by default when CUDA is available.

Notes:
- The launcher automatically sets common optimizations (TF32 for CUDA if available, cuDNN benchmark, etc.).
- Checkpoints are saved under `training.checkpoint_dir/<experiment_name>/`.

## 4. Evaluation
Evaluate any trained checkpoint:
```bash
python -m training.launch evaluate \
  --config config/base.yaml \
  --checkpoint checkpoints/exp_generic_gpu/best_model.pt \
  --test-data data/processed/test_final.txt \
  --batch-size 64 \
  --output-dir results
```
- Results are saved to `<output-dir>/evaluation_results.json`.

## 5. Profiling and Benchmarking
Profile a short training run and (optionally) benchmark common modes:
```bash
python -m training.launch profile \
  --config config/base.yaml \
  --profile-steps 20 \
  --benchmark \
  --output-dir profiling
```
- Generates traces and benchmark summaries for configs like mixed precision, large batch, compiled model, etc.

## 6. Progressive Training (Tiers)
- The progressive orchestrator groups languages into tiers and trains in stages.
- Entry point: `training/progressive_training.py` (programmatic orchestrator).
  - It will derive tier-specific configs, optionally resume from previous tier checkpoints, and record tier metrics.
- Typical usage pattern (Python API):
  ```python
  from training.progressive_training import ProgressiveTrainingOrchestrator

  orchestrator = ProgressiveTrainingOrchestrator(
      base_config_path='config/base.yaml',
      checkpoint_base_dir='checkpoints/progressive'
  )
  # Then call orchestrator methods to run tiers sequentially
  # (See the file for details; it prepares tier configs and spawns training runs.)
  ```

## 7. Bootstrapping From Pretrained (Optional)
Create initial encoder/decoder weights from pretrained models to warm-start training:
```bash
python -c "from training.bootstrap_from_pretrained import PretrainedModelBootstrapper as B; \
b=B(); b.create_encoder_from_pretrained('xlm-roberta-base', 'models/encoder/universal_encoder_initial.pt', 1024)"
```
- This produces `models/encoder/universal_encoder_initial.pt` (and similar for decoder if used).
- The launcher will automatically load these if present.

## 8. Quantization and Export (Post-Training)
Produce deployment-friendly variants (INT8, FP16, mixed):
```bash
python -c "from training.quantization_pipeline import EncoderQuantizer; \
q=EncoderQuantizer(); q.create_deployment_versions( \
  'checkpoints/exp_generic_gpu/best_model.pt', \
  calibration_data_path=None, \
  test_data_path='data/processed/test_final.txt' )"
```
- Outputs additional model files (e.g., `_int8.pt`, `_fp16.pt`) plus a comparison report.

## 9. Monitoring & Logs
- Logs directory: set via `--log-dir` (default `logs/`).
- GPU monitoring: `nvidia-smi` (Linux), Task Manager (Windows).
- Optional Weights & Biases: enable in config (`monitoring.use_wandb: true`).

## 10. Troubleshooting
- **Missing data files**: Ensure `data/processed/train_final.txt` and `val_final.txt` exist. Re-run `python -m data.unified_data_pipeline`.
- **Out of memory**: lower `training.batch_size`; enable/keep mixed precision; reduce sequence lengths; try multi-GPU with `--distributed`.
- **Slow training**: enable `torch.compile` via config; ensure latest CUDA drivers; increase DataLoader workers; verify GPU utilization.
- **Checkpoint load failures**: confirm path and integrity; use `training.training_validator.TrainingValidator.validate_checkpoint` to inspect.

---
- The old scripts `training/train_universal_system.py` and `training/distributed_train.py` are superseded by the unified launcher `python -m training.launch` with `train`, `evaluate`, and `profile` subcommands.