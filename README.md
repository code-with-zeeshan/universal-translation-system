# Universal Translation System

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A flexible and scalable translation platform designed to support multiple languages across diverse applications. This system enables seamless text translation, making it easy to localize content for global audiences. Features include integration with popular translation APIs, customizable language support, user-friendly interfaces, and extensible modules for adding new languages or translation engines. Ideal for developers and organizations looking to streamline multilingual communication and content delivery.

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
- ✅ CI/CD pipelines for encoder/decoder and SDKs

## 🎯 Usage Modes

You can use `universal-decoder-node` in two ways:

- **Personal Use:**  
  Run the decoder on your own device or cloud for private translation needs and testing. No registration is required.

- **Contributing Compute Power:**  
  If you want to support the project and make your node available to the global system, register your node (see below) so it can be added to the public decoder pool.

See [CONTRIBUTING.md](CONTRIBUTING.md) for registration instructions.

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/code-with-zeeshan/universal-translation-system
cd universal-translation-system

# Install dependencies
pip install -r requirements.txt

# Download sample data (for testing)
python data/download_sample_data.py

# Run local test
pytest tests/
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
```Javascript
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
- **Vocabulary Packs**: Downloadable language-specific token mappings
- **Model Weights**: Shared between all languages, trained on a diverse corpus

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## 📚 Documentation
- [API Documentation](docs/API.md)
- [Architecture Overview](docs/ARCHITECTURE.md)
- [Training Guide](docs/TRAINING.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [CI/CD Guide](docs/CI_CD.md)
- [SDK Integration Guide](docs/SDK_INTEGRATION.md)
- [Monitoring Guide](monitoring/README.md)
- [Contributing Guidelines](CONTRIBUTING.md)
- [License](LICENSE)

## 📊 Monitoring
- All services expose Prometheus metrics at `/metrics` (see [monitoring/README.md](monitoring/README.md))
- System metrics available on port 9000 if `system_metrics.py` is running
- Visualize with Grafana, set up alerts for latency, errors, and resource usage

## 🤝 Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and more details on how to contribute.

## ⚠️ Current Status
This is a research project in active development. Core components are implemented but not production-tested.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project builds upon amazing work from the ML community:

### Models & Research
- [NLLB-200](https://github.com/facebookresearch/fairseq/tree/nllb) by Meta AI - Inspiration and pretrained models
- [XLM-RoBERTa](https://huggingface.co/xlm-roberta-base) - Encoder initialization
- [mBART](https://github.com/pytorch/fairseq/tree/master/examples/mbart) - Decoder architecture insights

### Libraries & Tools
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Transformers](https://huggingface.co/transformers) by Hugging Face - Model implementations
- [ONNX Runtime](https://onnxruntime.ai/) - Mobile inference
- [SentencePiece](https://github.com/google/sentencepiece) - Tokenization
- [Litserve](https://github.com/litserve/litserve) - High-performance AI serving
- [Prometheus](https://prometheus.io/) & [Grafana](https://grafana.com/) - Monitoring

### Data Sources
- [OPUS](https://opus.nlpl.eu/) - Parallel corpora
- [Tatoeba](https://tatoeba.org/) - Community translations
- [FLORES-200](https://github.com/facebookresearch/flores) - Evaluation data

### Community
Special thanks to the open-source community for making projects like this possible.

## 📜 Changelog
See [CHANGELOG.md](CHANGELOG.md) for the latest updates.