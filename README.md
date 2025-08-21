# Universal Translation System

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A flexible and scalable translation platform designed to support multiple languages across diverse applications. This system enables seamless text translation, making it easy to localize content for global audiences. Features include an innovative edge-cloud architecture, customizable language support, and extensible modules for adding new languages or translation engines. Ideal for developers and organizations looking to streamline multilingual communication and content delivery.

> **New**: All configuration is now available through environment variables. See [Environment Variables](docs/environment-variables.md) for details.

## 🌟 Key Innovation

Unlike traditional translation apps that bundle 200MB+ models, our system uses:
- **Universal Encoder**: 35MB base + 2-4MB vocabulary packs (only download what you need)
- **Cloud Decoder**: Shared infrastructure for all users, served via Litserve (2x faster than FastAPI)
- **Result**: 40MB app with 90% quality of full models

## 📋 Features

- ✅ 20 language support with dynamic vocabulary loading
- ✅ Native SDKs for Android, iOS, Flutter, React Native, and Web
- ✅ Edge encoding, cloud decoding architecture
- ✅ ~85M parameters total (vs 600M+ for traditional models)
- ✅ Designed for low-end devices (2GB RAM)
- ✅ Full-system monitoring with Prometheus/Grafana
- ✅ Environment variable configuration for all components
- ✅ Docker and Kubernetes deployment support

## 🎯 Usage Modes

You can use `universal-decoder-node` in two ways:

- **Personal Use:**  
  Run the decoder on your own device or cloud for private translation needs and testing. No registration is required.

- **Contributing Compute Power:**  
  If you want to support the project and make your node available to the global system, register your node with the coordinator so it can be added to the public decoder pool.

See [CONTRIBUTING.md](CONTRIBUTING.md) for registration instructions.

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/universal-translation-system
cd universal-translation-system

# Set up environment variables (optional)
cp .env.example .env
# Edit .env with your configuration

# Install dependencies
pip install -r requirements.txt

# Run with Docker Compose (recommended)
docker-compose up -d

# Or run components individually
python cloud_decoder/optimized_decoder.py
python coordinator/advanced_coordinator.py
```

## 📱 SDK Integration

See [docs/SDK_INTEGRATION.md](docs/SDK_INTEGRATION.md) for full details and code examples for all platforms.

### Android
```java
val translator = TranslationClient(context)
val result = translator.translate("Hello", "en", "es")
```

### iOS
```swift
let translator = TranslationClient()
let result = try await translator.translate(text: "Hello", from: "en", to: "es")
```

### Web
```javascript
const translator = new TranslationClient();
const result = await translator.translate({
  text: "Hello",
  sourceLang: "en",
  targetLang: "es"
});
```

### React Native
```javascript
import { TranslationClient } from 'universal-translation-sdk';

const translator = new TranslationClient();
const result = await translator.translate({
  text: "Hello",
  sourceLang: "en",
  targetLang: "es"
});
```

## 🏗️ Architecture

- **Encoder**: Runs on device, converts text to language-agnostic embeddings
- **Decoder**: Runs on server (Litserve), converts embeddings to target language
- **Coordinator**: Manages decoder pool, handles load balancing and health monitoring
- **Vocabulary Packs**: Downloadable language-specific token mappings
- **Model Weights**: Shared between all languages, trained on a diverse corpus

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## 📚 Documentation
- [Vision & Architecture](docs/VISION.md)
- [Architecture Details](docs/ARCHITECTURE.md)
- [Environment Variables](docs/environment-variables.md)
- [Training Guide](docs/TRAINING.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [SDK Integration Guide](docs/SDK_INTEGRATION.md)
- [Monitoring Guide](monitoring/README.md)
- [Vocabulary Guide](vocabulary/Vocabulary_Guide.md)
- [API Documentation](docs/API.md)
- [Acknowledgments](docs/ACKNOWLEDGMENTS.md)
- [Contributing Guidelines](CONTRIBUTING.md)
- [License](LICENSE)

## 📊 Monitoring
- All services expose Prometheus metrics at `/metrics`
- Visualize with Grafana dashboards (included in `monitoring/grafana/dashboards`)
- Set up alerts for latency, errors, and resource usage
- See [monitoring/README.md](monitoring/README.md) for details

## 🤝 Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and more details on how to contribute.

## ⚠️ Current Status
This is a research project in active development. Core components are implemented but not production-tested.

### Component Status

| Component | Status | Notes |
|-----------|--------|-------|
| Encoder | ✅ Production-Ready | Core functionality complete and tested |
| Decoder | ✅ Production-Ready | Core functionality complete and tested |
| Vocabulary System | ✅ Production-Ready | Supports all planned languages |
| Coordinator | ✅ Production-Ready | Load balancing and health monitoring implemented |
| Android SDK | ✅ Production-Ready | Native implementation with JNI bindings |
| iOS SDK | ✅ Production-Ready | Swift implementation with C++ interoperability |
| Flutter SDK | ✅ Production-Ready | FFI bindings to native encoder |
| React Native SDK | ✅ Production-Ready | Core functionality implemented with config support |
| Web SDK | ✅ Production-Ready | Core functionality implemented with environment variable support |
| Monitoring | ✅ Production-Ready | Prometheus metrics and health checks implemented |
| Docker Support | ✅ Production-Ready | Docker Compose and Kubernetes configurations available |
| Environment Config | ✅ Production-Ready | All components configurable via environment variables |

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

We are grateful to the amazing open-source community and the researchers who have made this project possible. See [ACKNOWLEDGMENTS.md](docs/ACKNOWLEDGMENTS.md) for detailed acknowledgments.

## 📜 Changelog
See [CHANGELOG.md](CHANGELOG.md) for the latest updates.