# Developer Onboarding Guide

Welcome to the Universal Translation System! This guide will help you get started with development and understand the system architecture.

## Table of Contents
1. [System Overview](#system-overview)
2. [Getting Started](#getting-started)
3. [Development Environment Setup](#development-environment-setup)
4. [Key Components](#key-components)
5. [Common Workflows](#common-workflows)
6. [Troubleshooting](#troubleshooting)
7. [Resources](#resources)

## System Overview

The Universal Translation System uses an innovative edge-cloud split architecture:

- **Edge (Client)**: A lightweight universal encoder (35MB base + 2-4MB vocabulary packs) runs on the device
- **Cloud**: A powerful decoder infrastructure processes the encoded embeddings and returns translations

This approach allows us to deliver high-quality translations with minimal client-side resources.

### Key Features

- **Edge-Cloud Split Architecture**: Minimizes client app size while maximizing translation quality
- **Universal Encoder**: 35MB base model + 2-4MB vocabulary packs per language
- **Cloud Decoder**: Shared infrastructure using Litserve (2x faster than FastAPI)
- **Multiple SDK Support**: Native implementations for Android, iOS, Flutter, React Native, and Web
- **Dynamic Vocabulary System**: Download only the languages you need (2-4MB each)
- **Advanced Coordinator**: Load balancing, health monitoring, and dynamic decoder pool management

## Getting Started

### Prerequisites

- Python 3.8+
- Git
- Docker (optional, for containerized development)
- GPU with CUDA support (optional, for training and local decoder testing)

### Quick Start

```bash
# Clone repository
git clone https://github.com/code-with-zeeshan/universal-translation-system
cd universal-translation-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run system validation
python main.py validate

# Run tests
pytest tests/
```

## Development Environment Setup

### Python Environment

We recommend using a virtual environment for development:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### IDE Setup

We recommend using Visual Studio Code with the following extensions:
- Python
- Pylance
- Black Formatter
- Mermaid Preview
- Docker

### Docker Setup (Optional)

For containerized development:

```bash
# Build and run the encoder
docker build -t universal-encoder -f docker/encoder.Dockerfile .
docker run -p 8000:8000 universal-encoder

# Build and run the decoder (requires GPU)
docker build -t universal-decoder -f docker/decoder.Dockerfile .
docker run --gpus all -p 8001:8001 universal-decoder
```

## Key Components

### 1. Universal Encoder

The encoder converts text to language-agnostic embeddings. It's implemented in Python (for training) and C++ (for deployment).

**Key Files:**
- `encoder/`: Python implementation
- `encoder_core/`: C++ core implementation
- `vocabulary/`: Language-specific vocabulary management

### 2. Cloud Decoder

The decoder converts embeddings to translated text. It runs on GPU servers and is served via Litserve.

**Key Files:**
- `cloud_decoder/`: Server-side decoder implementation
- `universal-decoder-node/`: Standalone decoder node implementation

### 3. Coordinator

The coordinator manages communication between encoders and decoders, handling load balancing and health monitoring.

**Key Files:**
- `coordinator/`: Advanced routing system implementation
- `monitoring/`: Prometheus metrics collection and Grafana dashboards

### 4. SDKs

The system includes SDKs for multiple platforms:

**Key Directories:**
- `android/`: Android native SDK
- `ios/`: iOS native SDK
- `flutter/universal_translation_sdk/`: Flutter SDK
- `react-native/UniversalTranslationSDK/`: React Native SDK
- `web/universal-translation-sdk/`: Web SDK

## Common Workflows

### 1. Adding a New Language

To add support for a new language:

1. Prepare training data for the language pair
2. Update the vocabulary system:
   ```bash
   python vocabulary/create_vocabulary.py --lang <new_lang_code> --data <path_to_data>
   ```
3. Train the model with the new language:
   ```bash
   python main.py train --config config/training_new_lang.yaml
   ```
4. Update SDK vocabulary lists
5. Test the new language pair

### 2. Making Changes to the Encoder

1. Modify the encoder code in `encoder/` or `encoder_core/`
2. Run unit tests:
   ```bash
   pytest tests/test_encoder.py
   ```
3. Build the encoder core (if C++ changes were made):
   ```bash
   cd encoder_core
   mkdir build && cd build
   cmake ..
   make
   ```
4. Update the SDKs if the interface changed

### 3. Making Changes to the Decoder

1. Modify the decoder code in `cloud_decoder/`
2. Run unit tests:
   ```bash
   pytest tests/test_decoder.py
   ```
3. Build and test the decoder:
   ```bash
   python main.py benchmark --component decoder
   ```

### 4. Working with SDKs

Each SDK has its own build process:

**Android:**
```bash
cd android
./gradlew build
```

**iOS:**
```bash
cd ios
swift build
```

**Flutter:**
```bash
cd flutter/universal_translation_sdk
flutter pub get
flutter test
```

**React Native:**
```bash
cd react-native/UniversalTranslationSDK
npm install
npm test
```

**Web:**
```bash
cd web/universal-translation-sdk
npm install
npm run build
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   python scripts/check_dependencies.py
   ```

2. **CUDA Issues**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Vocabulary Loading Failures**
   ```bash
   python vocabulary/validate_vocabulary.py --lang <lang_code>
   ```

4. **Decoder Connection Issues**
   ```bash
   curl -v http://localhost:8001/health
   ```

For more detailed troubleshooting, see [TROUBLESHOOT.md](TROUBLESHOOT.md).

## Resources

- [Architecture Overview](ARCHITECTURE.md)
- [API Documentation](API.md)
- [Training Guide](TRAINING.md)
- [Deployment Guide](DEPLOYMENT.md)
- [SDK Integration Guide](SDK_INTEGRATION.md)
- [Troubleshooting Guide](TROUBLESHOOT.md)
- [Contributing Guidelines](../CONTRIBUTING.md)