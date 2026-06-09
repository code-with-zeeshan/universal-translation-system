# Runtime Filesystem Layout

All directories and files created by the system during end-to-end operation. Paths are relative to the project root unless noted.

## Quick Reference

```
project-root/
├── checkpoints/              Training checkpoints
├── config/                   Generated config + pool state
├── data/                     Data pipeline (raw → processed → final)
├── evaluation_reports/       Evaluation results
├── logs/                     Rotating log files
├── models/                   Model artifacts + version registry
├── training_visualizations/  Training dashboard PNGs
├── vocabulary/vocab/         SentencePiece packs + manifest
├── profiles/                 UDN profiling output
├── pipeline_state.json       Cross-stage auto-resume tracker
├── streaming_evaluation_cache.json
├── {phase}_checkpoint.json   Phase-level checkpoint (data/train/eval)
└── ~/.UniversalTranslationSystem/
    └── credentials.json      Encrypted credential store
```

---

## Data Pipeline

```
data/
├── log/                                        Pipeline log files
├── raw/
│   ├── {lang_pair}.txt                         Raw OPUS downloads
│   ├── mono_{lang}.txt                        Wikipedia monolingual data
│   └── opus/{pair}.txt                        Direct OPUS download
├── processed/
│   ├── corpus/{lang}_corpus.txt                 Monolingual corpora
│   ├── {pair}.txt                             Tiny dry-run samples
│   ├── train_final.txt                        Final training data
│   ├── val_final.txt                          Validation split
│   ├── train_temp.txt                         Temp file (replaced)
│   ├── cache/
│   │   ├── {stem}_cache.json                 Dataset cache metadata
│   │   └── {stem}_tokens_ml{max}_{src/tgt/mask}.npy  Tokenized memmap
│   ├── sampled/{pair}_sampled.txt             Smart-sampled data
│   ├── pivot_pairs/{src}-{tgt}_pivot.txt      Pivot translations
│   └── final/
│       ├── augmented_{pair}.txt               Backtranslation
│       ├── {src}_{tgt}/
│       │   ├── ff_{pair}.txt / ff_dynamic_{pair}.txt  False friends
│       │   └── idiom_{src}_{tgt}.txt          Idiom examples
│       ├── formal_{pair}.txt / casual_{pair}.txt  Register variants
│       ├── tone_{pair}.txt / bt_{pair}.txt    Backtranslation
│       ├── noised_{pair}.txt                  Noise augmentation
│       └── distilled/{pair}_distilled.txt     Knowledge-distilled
├── evaluation/
│   └── {pair}.tsv                            Eval test sets
└── pipeline_checkpoint.json                   Per-stage resume state
```

**Created by:** `data/pipeline_orchestrator.py`, `data/pipeline_state.py`, `data/connector/`

---

## Training

```
checkpoints/
├── {experiment_name}/
│   ├── best_model.pt                          Best validation loss
│   ├── emergency_checkpoint.pt                OOM/signal recovery
│   ├── training_report.json                   Epoch metrics summary
│   ├── {step}.pt                              Per-step checkpoint
│   ├── {step}_model.safetensors                Safetensors variant
│   └── {step}_metadata.pt                     Safetensors metadata
└── progressive/
    ├── progressive_state.json                 Multi-tier state
    ├── progressive_training_report.json       Final tier report
    ├── validation_config.yaml                 Validation config
    └── {tier}/
        └── best_model.pt                      Per-tier best model
```

**Created by:** `training/trainer.py`, `training/memory_trainer.py`, `training/progressive_training.py`

---

## Model Artifacts

```
models/
├── model_registry.json                        Version registry
├── .model_registry.lock                       File lock
├── encoder/
│   └── universal_encoder_initial.pt           Initialized encoder
├── decoder/
│   └── universal_decoder_initial.pt           Initialized decoder
├── adapters/
│   ├── {lang}_adapter.pt                      Per-language adapter
│   ├── best_{lang}_adapter.pt                 Best validation
│   ├── final_{lang}_adapter.pt                Final trained
│   ├── base_encoder_{int8,fp16,fp32}.pt       Quantized variants
│   └── {quant_mode}_encoder.pt                Quantization output
└── production/
    ├── encoder.pt                             Publish-ready encoder
    ├── decoder.pt                             Publish-ready decoder
    └── encoder.onnx                           ONNX export
```

**Created by:** `training/launch.py`, `training/train_adapters.py`, `training/quantization_pipeline.py`, `scripts/publish.py`

---

## Evaluation

```
evaluation_reports/
├── evaluation_report.json                     Overall summary
└── report_{test_file.stem}.json              Per-test-file results
```

**Created by:** `evaluation/evaluate_model.py`

---

## Logs

```
logs/
├── translation_system.log                     All components (rotating, 10×10MB)
├── errors.log                                 ERROR+ only (rotating, 5×10MB)
├── data/data.log                              Data pipeline
├── training/training.log                      Training run
├── monitoring/monitoring.log                  System metrics
├── coordinator/coordinator.log                Coordinator service
├── decoder/decoder.log                        Decoder service
├── evaluation/                                Evaluation output
└── vocabulary/vocabulary.log                  Vocabulary operations
```

**Created by:** `utils/logging_config.py` (sectioned logging handlers)

---

## Vocabulary

```
vocabulary/vocab/
├── manifest.json                              All pack metadata
├── {pack_name}.model                          SentencePiece model
├── {pack_name}_v{version}.msgpack             Verifiable pack file
├── {pack_name}_v{version}.json                Pack checksum + metadata
├── temp_{pack_name}.model                     SP training temp (deleted)
└── temp_{pack_name}.vocab                     SP vocab temp (deleted)
```

**Created by:** `vocabulary/vocab_production.py`, `vocabulary/vocab_validation.py`

---

## Serving / Coordinator

```
config/
├── decoder_pool.json                          Pool membership state
├── generated_config.yaml                       Config wizard output
└── generated_config.json

profiles/                                       UDN profiler
└── profile-{timestamp}.{format}

streaming_evaluation_cache.json                Streaming eval cache
```

**Created by:** `coordinator/advanced_coordinator.py`, `tools/register_decoder_node.py`, `config/config_wizard.py`, `udn/utils/profiler.py`

---

## Pipeline Auto-Resume State

```
pipeline_state.json                            Global data→train→eval tracker
data_checkpoint.json                           Data phase auto-resume
```

**Created by:** `utils/pipeline_checkpoint.py`

---

## Credentials (User Home)

```
~/.UniversalTranslationSystem/
└── credentials.json                           Encrypted PBKDF2-Fernet
```

**Created by:** `utils/credential_manager.py`

---

## Docker Volumes

| Container mount | Host path | Type |
|---|---|---|
| `/app/models` | `./models` | bind mount |
| `/app/vocabs` | `./vocabulary/vocab` | bind mount |
| `/app/config` | `./config` | bind mount |
| `/app/logs` | `./logs` | bind mount |
| `/app/.cache` | `decoder_cache` | named volume |
| `/var/lib/grafana` | `grafana-data` | named volume |
| `/data` | `redis-data` | named volume |

## Cleanup

```bash
# Remove all runtime data (keep code + config)
rm -rf checkpoints/ data/ logs/ models/ evaluation_reports/ profiles/
rm -f pipeline_state.json *_checkpoint.json streaming_evaluation_cache.json
rm -rf ~/.UniversalTranslationSystem/

# Full reset including generated configs
rm -rf config/decoder_pool.json config/generated_*
```
