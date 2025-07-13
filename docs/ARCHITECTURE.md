# System Architecture

## Overview

The Universal Translation System uses a split architecture to minimize mobile app size while maintaining translation quality.

## Components

### 1. Universal Encoder (Mobile)
- **Size**: 35MB base model
- **Architecture**: 6-layer transformer, 384 hidden dimensions
- **Vocabulary**: Dynamic loading (2-4MB per language pack)
- **Output**: Compressed embeddings (2-3KB per translation)

### 2. Vocabulary Packs
- **Latin Pack**: ~3MB (covers 12 languages)
- **CJK Pack**: ~4MB (Chinese, Japanese, Korean)
- **Arabic Pack**: ~2MB
- **Other Packs**: 1-2MB each

### 3. Universal Decoder (Server)
- **Size**: ~50M parameters
- **Architecture**: 6-layer transformer with cross-attention
- **Infrastructure**: Runs on GPU servers (T4 or better)

## Data Flow

1. User inputs text
2. App loads relevant vocabulary pack (if needed)
3. Encoder converts text → embeddings
4. Embeddings compressed and sent to server (2-3KB)
5. Decoder generates translation
6. Translation sent back to app

## Key Design Decisions

### Why Split Architecture?
- Mobile apps stay small (<50MB total)
- Expensive computation happens on server
- Users only download languages they need

### Why Universal Encoder?
- One encoder works with any vocabulary
- No need for language-specific models
- Enables zero-shot translation

### Vocabulary Management
- Packs are grouped by script similarity
- Common tokens shared across languages
- Subword tokenization for unknown words

## Model Specifications

### Encoder
```python
EncoderConfig = {
    'num_layers': 6,
    'hidden_dim': 384,
    'num_heads': 8,
    'ffn_dim': 1536,
    'max_seq_length': 128,
    'dynamic_vocab_size': 50000
}
```

### Decoder
```python
DecoderConfig = {
    'num_layers': 6,
    'hidden_dim': 512,
    'num_heads': 8,
    'ffn_dim': 2048,
    'max_seq_length': 256,
    'vocab_size': 50000
}
```

### Performance Characteristics

**Encoding Latency**: 10-50ms on mobile CPU
**Network Transfer**: 2-3KB per request
**Decoding Latency**: 20-50ms on server GPU
**Total Translation Time**: 100-200ms (including network)

📁 Complete Universal Translation System Directory Structure
🗂️ Full Project Structure

```bash
universal-translation-system/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── .env.example
├── CONTRIBUTING.md
├── LICENSE
├── CHANGELOG.md
├── checkpoints
│
├── data/
│   ├── essential
│   ├── processed
│   ├── raw
│   ├── __init__.py
│   ├── download_training_data.py
│   ├── download_curated_data.py
│   ├── smart_data_downloader.py
│   ├── smart_sampler.py
│   ├── synthetic_augmentation.py
│   └── practical_data_pipeline.py
│
├── vocabulary/
│   ├── __init__.py
│   ├── create_vocabulary_packs_from_data.py
│   └── vocab_cache/
│
├── training/
│   ├── __init__.py
│   ├── bootstrap_from_pretrained.py
│   ├── train_universal_system.py
│   ├── train_universal_models.py
│   ├── distributed_train.py
│   ├── memory_efficient_training.py
│   └── convert_models.py
│
├── encoder_core/
│   ├── CMakeLists.txt
│   ├── include/
│   │   └── universal_encoder.h
│   └── src/
│       └── universal_encoder.cpp
│
├── logs
├── models/
│   ├── encoder/
│   ├── decoder/
│   └── production/
│
├── android/
│   └── UniversalTranslationSDK/
│       ├── build.gradle
│       └── src/main/java/com/universaltranslation/encoder/
│           └── TranslationEncoder.kt
│
├── ios/
│   └── UniversalTranslationSDK/
│       └── Sources/
│           └── TranslationEncoder.swift
│
├── flutter/
│   └── universal_translation_sdk/
│       ├── pubspec.yaml
│       └── lib/src/
│           └── translation_encoder.dart
│
├── react-native/
│   └── UniversalTranslationSDK/
│       ├── package.json
│       └── src/
│           └── index.tsx
│
├── monitoring/
│   └── metrics_collector.py
│
├── web/
│   └── universal-translation-sdk/
│       ├── package.json
│       └── src/
│           └── index.ts
│
├── cloud_decoder/
│   ├── __init__.py
│   ├── optimized_decoder.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── docker/
│   ├── encoder.Dockerfile
│   ├── decoder.Dockerfile
│   └── docker-compose.yml
│
├── config/
│   ├── training_a100.yaml
│   ├── training_rtx3090.yaml
│   └── training_v100.yaml
│
├── vocabulary/
│   ├── __init__.py
│   ├── create_vocabulary_packs_from_data.py
│   └── vocab_cache/
│
├── kubernetes/
│   ├── namespace.yaml
│   ├── decoder-deployment.yaml
│   ├── decoder-service.yaml
│   └── coordinator-deployment.yaml
│
├── tests/
│   ├── __init__.py
│   ├── test_local.py
│   ├── test_encoder.py
│   ├── test_decoder.py
│   └── test_translation_quality.py
│
├── tools/
│   └── create_vocabulary_packs.py
│
├── scripts/
│   ├── train_from_scratch.sh
│   ├── build_models.py
│   ├── multi-GPU_training.sh
│   ├── deploy.sh
│   └── setup_environment.sh
│
└── docs/
    ├── API.md
    ├── Architecture.md
    ├── DEPLOYMENT.md
    ├── Roadmap.md
    ├── TROUBLESHOOT.md
    └── TRAINING.md
```