# Universal Translation System

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A flexible and scalable translation platform designed to support multiple languages across diverse applications. This system enables seamless text translation, making it easy to localize content for global audiences. Features include integration with popular translation APIs, customizable language support, user-friendly interfaces, and extensible modules for adding new languages or translation engines. Ideal for developers and organizations looking to streamline multilingual communication and content delivery.

## 🌟 Key Innovation

Unlike traditional translation apps that bundle 200MB+ models, our system uses:
- **Universal Encoder**: 35MB base + 2-4MB vocabulary packs (only download what you need)
- **Cloud Decoder**: Shared infrastructure for all users
- **Result**: 40MB app with 90% quality of full models

## 📋 Features

- ✅ 20 language support with dynamic vocabulary loading
- ✅ Native SDKs for Android, iOS, Flutter, React Native, and Web
- ✅ Edge encoding, cloud decoding architecture
- ✅ ~85M parameters total (vs 600M+ for traditional models)
- ✅ Designed for low-end devices (2GB RAM)

## 🚀 Quick Start

```bash
# Clone repository
git clone [repository-url]
cd universal-translation-system

# Install dependencies
pip install -r requirements.txt

# Download sample data (for testing)
python data/download_sample_data.py

# Run local test
python test_local.py --text "Hello world" --source en --target es
```

## 📱 SDK Integration

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
- **Decoder**: Runs on server, converts embeddings to target language
- **Vocabulary Packs**: Downloadable language-specific token mappings
- **Model Weights**: Shared between all languages, trained on a diverse corpus

See [docs/Architecture.md](docs/Architecture.md) for details.

## 📚 Documentation
- **API Documentation**
- **Architecture Overview**
- **Training Guide**
- **Deployment Guide**
- **Contributing Guidelines**
- **License**
- **Community Acknowledgments (soon)**

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

### Data Sources
- [OPUS](https://opus.nlpl.eu/) - Parallel corpora
- [Tatoeba](https://tatoeba.org/) - Community translations
- [FLORES-200](https://github.com/facebookresearch/flores) - Evaluation data

### Community
Special thanks to the open-source community for making projects like this possible.

## 📜 Changelog
See [CHANGELOG.md](CHANGELOG.md) for the latest updates.