# Vocabulary Pack Creation Scripts Guide

## 📋 Overview

The Universal Translation System uses a config-driven, orchestrated pipeline for vocabulary management. Vocabulary packs are created, registered, and integrated with the coordinator and SDKs for seamless language support and monitoring.

## 🔧 Available Scripts

### Script 1: `vocabulary/create_vocabulary_packs.py`
**Advanced Vocabulary Pack Creator with Corpus Analysis**

- Use for research, custom optimization, or domain-specific vocabularies
- Integrates with the config-driven pipeline (see `data/config.yaml`)

### Script 2: `vocabulary/create_vocabulary_packs_from_data.py`
**Production-Ready Vocabulary Pack Creator using SentencePiece**

- Use for standard, production deployments
- Automatically updates vocabulary packs and registers them in the system

## 🎯 Use Case Comparison

- Choose the script based on your need for customization, speed, and integration with the orchestrated pipeline
- All packs are referenced in `data/config.yaml` and used by the encoder, decoder, and SDKs

## 📊 Feature Comparison

- Both scripts support config-driven language groups and can be monitored via Prometheus metrics
- Packs are dynamically loaded by the encoder and registered with the coordinator for cloud decoding

## 🚀 Quick Start Guide

- Update `data/config.yaml` with new languages or groups
- Run the appropriate script to generate packs
- Packs are automatically integrated with the system and visible in the coordinator dashboard

## 💡 Real-World Examples

- See [docs/SDK_INTEGRATION.md](docs/SDK_INTEGRATION.md) for how SDKs use vocabulary packs
- Use the coordinator dashboard to monitor vocabulary usage and health

## 🤝 Contributing

- Add new scripts or improvements to support more languages, better compression, or advanced analytics
- Document your changes and update the config as needed

---

For more details, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/TRAINING.md](docs/TRAINING.md).