# Contributing to Universal Translation System

Thank you for your interest in contributing!

## How to Contribute

### Reporting Issues
- Check existing issues first
- Include steps to reproduce
- Include error messages and logs
- Specify your environment (OS, Python version, GPU)

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Ensure all tests pass (`pytest tests/`)
6. Commit with clear message
7. Push to your fork
8. Open a Pull Request

### Code Style

- Follow PEP 8 for Python code
- Use type hints where possible
- Add docstrings to functions and classes
- Run `black .` for formatting Python code
- Use ESLint for JavaScript/TypeScript code

### Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_encoder.py

# Run with coverage
pytest --cov=. tests/

# Web SDK
cd web/universal-translation-sdk && npm install && npm test

# React Native SDK
cd react-native/UniversalTranslationSDK && npm install && npm test

# Android SDK (unit tests)
cd android/UniversalTranslationSDK && ./gradlew test

# iOS SDK (SwiftPM tests)
cd ios/UniversalTranslationSDK && swift test
```

### Publishing (SDKs)
- See docs/SDK_PUBLISHING.md for publishing Android (Maven), iOS (CocoaPods/SPM), and RN linking.
- Web SDK publishing to npm is available via GitHub Actions (web-npm-publish.yml) with NPM_TOKEN secret.

### Areas We Need Help
1. **Language Support**: Adding more languages and improving vocabulary packs
2. **Mobile SDKs**: Enhancing iOS/Android/Flutter/React Native implementations
3. **Optimization**: Making models smaller/faster through quantization techniques
4. **Documentation**: Improving guides and examples
5. **Testing**: Adding more test cases and integration tests

### Development Setup

#### Using Docker (Recommended)
```bash
# Clone repo
git clone https://github.com/yourusername/universal-translation-system
cd universal-translation-system

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your configuration

# Start services with Docker Compose
docker-compose up -d
```

#### Manual Setup
```bash
# Clone repo
git clone https://github.com/yourusername/universal-translation-system
cd universal-translation-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Run components individually
python cloud_decoder/optimized_decoder.py
python coordinator/advanced_coordinator.py
```

### Contributing Compute Resources

#### Usage Modes

There are two ways to run a decoder node (`universal-decoder-node`):

- **Private/Personal Use:**  
  You may run a decoder node for your own needs or testing without registering it with the project. No further action is needed.
  > Run the decoder on your own device or cloud for private translation needs and testing. Registration is NOT required if you do not wish to contribute compute power to the project.

- **Public Contribution:**  
  If you wish to contribute compute power to support the Universal Translation System, register your node so it can be added to the global decoder pool and serve requests for all users.

Want to help by running a decoder node? Here's how:

#### Option 1: Quick Deploy (Managed)
```bash
# Install the universal-decoder-node package
pip install universal-decoder-node

# Register your node with the coordinator
universal-decoder-node register --name "your-node-name" --endpoint "https://your-decoder.com" --gpu-type "T4" --capacity 100 --coordinator-url "https://coordinator.example.com"

# Start the decoder service
universal-decoder-node start --host 0.0.0.0 --port 8000 --workers 4
```

#### Option 2: Custom Deploy with Docker
1. Clone the repository
2. Configure environment variables in `.env` file
3. Run with Docker Compose:
   ```bash
   docker-compose up -d decoder
   ```
4. Register your node with the coordinator:
   ```bash
   docker-compose exec decoder universal-decoder-node register --name "your-node-name" --endpoint "https://your-decoder.com"
   ```

**Decoder Node Requirements**
- HTTPS endpoint with valid certificate (for production)
- GPU support recommended (T4, V100, A100, etc.)
- Minimum 8GB RAM, 16GB recommended
- Support for health checks (`/health` endpoint)

**Benefits for Contributors**
- Recognition in project contributors list
- Access to the coordinator dashboard
- Priority support for issues

### How Encoder and Decoder Communicate
- **Encoder (Edge/SDK)** encodes text to embeddings on device
- **Decoder (Cloud Node)** exposes a REST API (e.g., `/decode` endpoint, served by Litserve)
- **Communication Protocol**: The encoder sends compressed embeddings (binary) to the decoder's `/decode` endpoint, specifying the target language in the header. The decoder returns the translated text as JSON.
- **Coordinator** manages the decoder pool, handles load balancing, and monitors health
- See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/API.md](docs/API.md) for protocol details.

## 🌍 Our Mission

To break down language barriers by making high-quality translation accessible on every device, regardless of internet speed or hardware limitations.

We believe communication is a human right, not a luxury.

## 📊 Project Health

![Languages Supported](https://img.shields.io/badge/languages-20-brightgreen)
![Model Size](https://img.shields.io/badge/encoder%20size-35MB-blue)
![Decoder Size](https://img.shields.io/badge/decoder%20size-350MB-blue)

## Community & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/universal-translation-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/universal-translation-system/discussions)

**[Code of Conduct](CODE_OF_CONDUCT.md)**
Please be respectful and constructive in all interactions.