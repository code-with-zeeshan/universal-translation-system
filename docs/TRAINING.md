# Training Guide

## Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 50GB+ free disk space
- 16GB+ RAM

## 1. Data Preparation
1. Run the unified data pipeline:
   ```bash
   python -m data.pipeline_orchestrator
   ```
   - Or using the unified pipeline CLI:
   ```bash
   python scripts/pipeline.py data --config config/base.yaml
   ```
   - Processed files are expected under `data/processed/` (e.g., `train_final.txt`, `val_final.txt`, `test_final.txt`).
   - The dataset initializes `UnifiedVocabularyManager` using your config.

2. Create or update vocabulary packs:
   ```bash
   # Using the pipeline CLI
   python scripts/pipeline.py vocab --mode production --corpus-dir ./data/processed --output-dir ./vocabs
   ```
   - Configure vocabulary dir: `vocabulary.vocab_dir` in config.
   - See [Vocabulary_Guide.md](Vocabulary_Guide.md) for details.

## 2. Configuration
- Base config: `config/base.yaml`
- All config models defined in `config/schemas.py` (canonical hierarchy, merged from `config_models.py`).
- Key fields used by the launcher:
  - **data.processed_dir**: path containing `train_final.txt`, `val_final.txt`, `test_final.txt`
  - **data.cache_dir**: optional dataset cache
  - **vocabulary.vocab_dir**: directory with vocabulary packs
  - **model**: `vocab_size`, `hidden_dim`, `num_layers`, `num_heads`, `decoder_dim`, `decoder_layers`, `decoder_heads`, `dropout`
  - **training**: `batch_size`, `learning_rate`, `num_epochs`, `checkpoint_dir`
  - **monitoring**: `use_wandb` etc.

## 3. Training (Intelligent Trainer)
Use the consolidated launcher with subcommands. The `IntelligentTrainer` class lives in `training/trainer.py`.

- Basic single-GPU/CPU training:
  ```bash
  python -m training.launch train --config config/base.yaml
  ```

- Hardware-aware presets (configured via `training/hardware_profile.py`):
  ```bash
  # Examples (all use config/base.yaml with hardware-specific overrides)
  python -m training.launch train --config config/base.yaml
  ```

- Override common hyperparameters from CLI:
  ```bash
  python -m training.launch train \
    --config config/base.yaml \
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
    --config config/base.yaml \
    --distributed \
    --experiment-name exp_multi
  ```

- Evaluation:
  ```bash
  python -m training.launch evaluate --config config/base.yaml --checkpoint checkpoints/exp_generic_gpu/best_model.pt
  ```

Notes:
- The launcher automatically sets TF32 for CUDA if available, cuDNN benchmark, etc.
- Checkpoints are saved under `training.checkpoint_dir/<experiment_name>/`.
- Memory optimization via `training/memory_trainer.py` (MemoryOptimizedTrainer) is activated automatically when memory mode is enabled.

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
- See `evaluation/evaluator.py` and `evaluation/metrics.py` for details.

## 5. Profiling and Benchmarking
Profile a short training run and (optionally) benchmark common modes:
```bash
python -m training.launch profile \
  --config config/base.yaml \
  --profile-steps 20 \
  --benchmark \
  --output-dir profiling
```
- See `training/training_analytics.py` and `training/model_profiler.py`.

## 6. Progressive Training (Tiers)
- Entry point: `training/progressive_training.py` (programmatic orchestrator).
```python
from training.progressive_training import ProgressiveTrainingOrchestrator

orchestrator = ProgressiveTrainingOrchestrator(
    base_config_path='config/base.yaml',
    checkpoint_base_dir='checkpoints/progressive'
)
```

## 7. Bootstrapping From Pretrained (Optional)
```bash
python -c "from training.bootstrap_from_pretrained import PretrainedModelBootstrapper as B; b=B(); b.create_encoder_from_pretrained('xlm-roberta-base', 'models/encoder/universal_encoder_initial.pt', 1024)"
python -c "from training.bootstrap_from_pretrained import PretrainedModelBootstrapper as B; b=B(); b.create_decoder_from_mbart('facebook/mbart-large-50', 'models/decoder/universal_decoder_initial.pt')"
```
- Produces: `models/encoder/universal_encoder_initial.pt`, `models/decoder/universal_decoder_initial.pt`
- The training launcher automatically loads these if present.

## 8. Quantization and Export (Post-Training)
Produce deployment-friendly variants (INT8, FP16, mixed) and optional ONNX/CoreML/TFLite conversions.

- Quantize encoder:
```bash
python -c "from training.encoder_quantizer import EncoderQuantizer; q=EncoderQuantizer(); q.create_deployment_versions('checkpoints/exp_generic_gpu/best_model.pt', test_data_path='data/processed/test_final.txt')"
```
- See `training/encoder_quantizer.py`, `training/quality_comparator.py`, `training/quantization_common.py`.

- Convert encoder to ONNX:
```bash
python -c "import torch; from training.convert_models import ModelConverter as C; dummy=torch.randint(0,50000,(1,128)); C.pytorch_to_onnx('checkpoints/exp_generic_gpu/best_model.pt','models/export/encoder.onnx', dummy)"
```
- Convert to CoreML or TFLite:
```bash
python -c "from training.convert_models import ModelConverter as C; C.onnx_to_coreml('models/export/encoder.onnx','models/export/encoder.mlpackage')"
python -c "from training.convert_models import ModelConverter as C; C.onnx_to_tflite('models/export/encoder.onnx','models/export/encoder.tflite')"
```

## 9. Monitoring & Logs
- Logs directory: set via `--log-dir` (default `logs/`).
- GPU monitoring: `nvidia-smi` (Linux).
- Optional Weights & Biases: enable in config (`monitoring.use_wandb: true`, default `false`).
- Metrics via `monitoring/metrics.py` and `monitoring/metrics_collector.py`.

## 10. Troubleshooting
- **Missing data files**: Ensure `data/processed/train_final.txt` and `val_final.txt` exist.
- **Out of memory**: lower `training.batch_size`; enable mixed precision; reduce sequence lengths; try `--distributed`.
- **Slow training**: enable `torch.compile` via config; ensure latest CUDA drivers; increase DataLoader workers.
- **Checkpoint load failures**: confirm path and integrity; use `training.training_validator.TrainingValidator.validate_checkpoint`.

---

- The old scripts `training/train_universal_system.py` and `training/distributed_train.py` are superseded by `python -m training.launch`.
- Backward-compatible shims exist for the split modules: `training.intelligent_trainer` re-exports from `training.trainer`, etc.
