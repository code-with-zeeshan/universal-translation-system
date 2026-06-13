# Runtime Filesystem Layout

All directories and files created by the system during end-to-end operation. Paths are relative to the project root unless noted.

> **All runtime paths are managed by `RuntimeDirectoryManager`** (`utils.common_utils.RuntimeDirectoryManager`). Every module that reads or writes runtime artifacts should obtain paths through this class. The layout below shows the default locations; they can be customised by passing a `RootConfig` with different field values to RDM.

## Quick Reference

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
│   └── augment/
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

**Created by:** `pipeline/data/orchestrator.py`, `pipeline/data/state.py`, `pipeline/connectors/data.py`, `pipeline/connectors/filter.py`

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

**Created by:** `pipeline/training/trainer.py`, `pipeline/training/memory/trainer.py`, `pipeline/training/progressive.py`, `pipeline/training/launch.py`

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

**Created by:** `pipeline/training/launch.py`, `pipeline/training/peft.py`, `pipeline/training/quantization/pipeline.py`, `pipeline/training/model_init.py`, `scripts/publish.py`

---

## Evaluation

```
evaluation_reports/
├── evaluation_report.json                     Overall summary
└── report_{test_file.stem}.json              Per-test-file results
```

**Created by:** `evaluation/evaluate_model.py`, `evaluation/evaluator.py`

---

## Logs

```
logs/
├── universal_translation_system.log           All components (rotating, 10×10MB)
├── errors.log                                 All ERROR+ (rotating, 5×10MB)
├── data/
│   ├── data.log                               Data pipeline INFO+
│   └── error.log                              Data pipeline ERROR+
├── training/
│   ├── training.log                           Training run DEBUG+
│   └── error.log                              Training run ERROR+
├── monitoring/
│   ├── monitoring.log                         System metrics INFO+
│   └── error.log                              System metrics ERROR+
├── coordinator/
│   ├── coordinator.log                        Coordinator INFO+
│   └── error.log                              Coordinator ERROR+
├── decoder/
│   ├── decoder.log                            Decoder INFO+
│   └── error.log                              Decoder ERROR+
├── evaluation/
│   ├── evaluation.log                         Evaluation INFO+
│   └── error.log                              Evaluation ERROR+
└── vocabulary/
    ├── vocabulary.log                         Vocabulary INFO+
    └── error.log                              Vocabulary ERROR+
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

**Created by:** `pipeline/vocabulary/production.py`, `pipeline/vocabulary/validation.py`, `pipeline/vocabulary/build.py`

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

**Created by:** `runtime/coordinator/advanced_coordinator.py`, `tools/register_decoder_node.py`, `scripts/config_wizard.py`, `evaluation/evaluator.py`, `universal-decoder-node/udn/utils/profiler.py`

---

## Pipeline Auto-Resume State

```
pipeline_state.json                            Global data→train→eval tracker
data_checkpoint.json                           Data phase auto-resume
```

**Created by:** `utils/pipeline_checkpoint.py`

---

## Scripts / Build Artifacts

```
scripts/
└── hf_upload.log                              Hugging Face upload log
```

Also creates temporary model copies during publish/ONNX export under `models/production/`.

**Created by:** `scripts/build_and_upload_pipeline.py`, `scripts/first_time_success.py`

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
rm -rf checkpoints/ data/ logs/ models/ evaluation_reports/ profiles/ training_visualizations/ vocabulary/
rm -f pipeline_state.json *_checkpoint.json streaming_evaluation_cache.json
rm -rf ~/.UniversalTranslationSystem/

# Full reset including generated configs
rm -rf config/decoder_pool.json config/generated_*
```
