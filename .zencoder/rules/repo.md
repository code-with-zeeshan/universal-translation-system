---
description: Repository Information Overview
alwaysApply: true
---

# Universal Translation System Information

## Summary
A flexible and scalable translation platform designed to support multiple languages across diverse applications. The system uses an innovative edge encoding, cloud decoding architecture with a universal encoder (35MB base + 2-4MB vocabulary packs) and cloud decoder infrastructure, resulting in a 40MB app with 90% quality of full models. The key innovation is splitting the translation process between lightweight client-side encoding and powerful server-side decoding. Version 1.0.0 includes enhanced dependency management, improved build processes, and better logging.

## Key Features
- **Edge-Cloud Split Architecture**: Minimizes client app size while maximizing translation quality
- **Universal Encoder**: 35MB base model + 2-4MB vocabulary packs per language
- **Cloud Decoder**: Shared infrastructure using Litserve (2x faster than FastAPI)
- **Multiple SDK Support**: Native implementations for Android, iOS, Flutter, React Native, and Web
- **Dynamic Vocabulary System**: Download only the languages you need (2-4MB each)
- **Advanced Coordinator**: Load balancing, health monitoring, and dynamic decoder pool management
- **Comprehensive Monitoring**: Prometheus/Grafana integration for system metrics
- **Optimized for Low-End Devices**: Works on devices with as little as 2GB RAM

## Structure
- **encoder**: Python-based universal encoder implementation with RoPE and SwiGLU optimizations
- **encoder_core**: C++ core implementation of the encoder for native platforms (FFI-compatible)
- **cloud_decoder**: Server-side decoder implementation with optimized memory usage for GPUs
- **vocabulary**: Language-specific vocabulary management with multiple operating modes (FULL, OPTIMIZED, EDGE)
- **web/universal-translation-sdk**: Web SDK implementation using TypeScript with npm packaging
- **react-native/UniversalTranslationSDK**: React Native SDK with native bridges for Android and iOS
- **flutter/universal_translation_sdk**: Flutter SDK using FFI for native performance across platforms
- **android**: Android native SDK implementation with JNI bindings to C++ encoder core
- **ios**: iOS native SDK using Swift with C++ interoperability
- **docker**: Docker configuration for encoder and decoder services with multi-stage builds
- **kubernetes**: Kubernetes deployment configurations for scalable cloud infrastructure
- **coordinator**: Advanced routing system for managing decoder pools with health monitoring
- **monitoring**: Prometheus metrics collection, health monitoring, and Grafana dashboards
- **training**: Training scripts and utilities for model development, including distributed training
- **evaluation**: Model evaluation and quality assessment tools with BLEU score calculation
- **tools**: Utility scripts for development, deployment, and node registration
- **tests**: Comprehensive test suite for all components with integration tests
- **docs**: Detailed documentation for architecture, APIs, deployment, and SDK integration
- **universal-decoder-node**: Standalone decoder node implementation for distributed computing
- **utils**: Shared utility functions for authentication, security, and resource monitoring
- **config**: Configuration schemas and default settings for all components
- **scripts**: Build and deployment scripts for automation

## Language & Runtime
**Languages**: Python 3.8+, C++17, TypeScript, Dart, Java, Swift
**Python Version**: 3.8+
**Build Systems**: CMake (C++), npm (Web/React Native), Flutter, Gradle (Android), Swift Package Manager (iOS)
**Package Managers**: pip (Python), npm (JavaScript), pub (Flutter)

## Dependencies
**Main Dependencies**:
- PyTorch 2.0.0+: Deep learning framework for model implementation
- Transformers 4.21.0+: Provides transformer architecture components
- ONNX Runtime 1.16.0+: Cross-platform inference acceleration
- LitServe 0.2.0+: High-performance model serving
- sentencepiece 0.2.0+: Tokenization for multiple languages
- msgpack 1.0.0+: Efficient binary serialization
- lz4 4.3.2+: Fast compression for embeddings
- zstandard: Advanced compression for vocabulary packs
- aiofiles: Asynchronous file operations
- opentelemetry: Distributed tracing and monitoring
- triton_python_backend_utils: GPU optimization utilities
- redis 5.0.0+: Caching and message queue support

**Development Dependencies**:
- pytest 7.4.0+: Testing framework
- black 23.10.0+: Code formatting
- mypy 1.6.0+: Static type checking
- pre-commit 3.5.0+: Git hooks for code quality
- jest: JavaScript testing
- flutter_test: Flutter testing utilities

## Architecture Details
The system uses a split architecture with:

### 1. Universal Encoder (Edge/Client)
- Runs on device or via API
- Converts text to language-agnostic embeddings
- Implemented in Python (training) and C++ (deployment)
- Optimized with RoPE (Rotary Position Embedding) and SwiGLU activations
- Supports dynamic vocabulary loading
- Memory-efficient with quantization support

### 2. Vocabulary System
- Multiple operating modes (FULL, OPTIMIZED, EDGE)
- Language-specific vocabulary packs (2-4MB each)
- Latin Pack (~3MB) covers 12 languages
- CJK Pack (~4MB) for Chinese, Japanese, Korean
- Bloom filters and prefix trees for edge optimization
- LRU caching for efficient memory usage

### 3. Cloud Decoder
- Runs on GPU servers (T4, 3090, V100, A100, or generic GPU)
- 6-layer transformer with cross-attention
- Served via Litserve for high performance
- Supports dynamic adapter loading for language-specific tuning
- Asynchronous processing for high concurrency
- Hugging Face Hub integration for model distribution
- Enhanced logging with timestamped output

### 4. Advanced Coordinator
- Load balancing across decoder pool using least-loaded routing algorithm
- Health monitoring with background thread checking decoder `/health` endpoints
- Dynamic decoder pool management via REST API and web dashboard
- JWT-based authentication and security features
- Web dashboard with real-time charts using Chart.js
- Prometheus metrics for monitoring system health
- Automatic failover when decoders become unhealthy
- Support for adding/removing decoders at runtime without downtime

## Build & Installation
```bash
# Python core
pip install -r requirements.txt

# Web SDK
cd web/universal-translation-sdk
npm install
npm run build:wasm  # Build WebAssembly components
npm run build

# React Native SDK
cd react-native/UniversalTranslationSDK
npm install
npm run prepare

# Flutter SDK
cd flutter/universal_translation_sdk
flutter pub get

# C++ encoder core
cd encoder_core
mkdir build && cd build
cmake ..
make
```

## Docker
**Encoder Dockerfile**: docker/encoder.Dockerfile
**Decoder Dockerfile**: docker/decoder.Dockerfile (includes wget for downloading models)
**Configuration**: docker-compose.yml with separate services for encoder and decoder

## Testing
**Framework**: pytest (Python), Jest (JavaScript/TypeScript), XCTest (iOS)
**Test Location**: tests/ directory for Python, src/__tests__ for JS/TS
**Run Command**:
```bash
# Python tests
pytest tests/

# Web SDK tests
cd web/universal-translation-sdk
npm test

# React Native tests
cd react-native/UniversalTranslationSDK
npm test

# C++ tests
cd encoder_core/build
ctest
```

## SDK Implementations

### Android SDK
- Native JNI bindings to C++ encoder core
- Dynamic vocabulary pack loading
- Efficient memory management with JNI references
- Background threading for non-blocking UI
- Fallback to API-based encoding when native fails
- Comprehensive error handling and reporting
- Support for Android 6.0 (API 23) and higher

### iOS SDK
- Swift implementation with C++ interoperability
- Swift Package Manager and CocoaPods support
- Memory-efficient vocabulary management
- Background processing with GCD
- Objective-C compatibility layer
- Comprehensive error handling with Swift Result type
- Support for iOS 13.0 and higher

### Flutter SDK
- FFI bindings to native C++ encoder
- Cross-platform implementation (iOS, Android, macOS, Windows, Linux)
- Asynchronous API with Futures
- Proper resource cleanup with finalizers
- Comprehensive error handling with sealed classes
- Memory usage monitoring
- Automatic vocabulary pack management

### React Native SDK
- JavaScript API with TypeScript definitions
- Native modules for Android and iOS
- Fallback to cloud-based encoding
- Promise-based async API
- Event-based progress reporting
- Comprehensive error handling
- Support for Expo and bare React Native projects

### Web SDK
- Pure TypeScript implementation
- WebAssembly support for browser-based encoding with custom build process
- Fallback to cloud-based encoding
- Promise-based async API
- Comprehensive error handling
- Bundle size optimization
- Support for all modern browsers
- Custom WASM build and copy scripts

## Data Flow
1. User inputs text in source language
2. App loads relevant vocabulary pack if needed
3. Encoder converts text to embeddings (on device or via API)
4. Embeddings are compressed with LZ4 and sent to coordinator
5. Coordinator routes request to least-loaded decoder
6. Decoder generates translation in target language
7. Translation is returned to the app with confidence score

## Monitoring
The system includes comprehensive monitoring:
- Prometheus metrics for all components
- Health checks for decoders and coordinator
- Resource usage tracking (CPU, GPU, memory)
- Request latency and throughput metrics
- Error rate monitoring
- Vocabulary usage analytics
- Grafana dashboards for visualization
- Enhanced logging with timestamped output and log file rotation
- Dependency checking with fallback mechanisms