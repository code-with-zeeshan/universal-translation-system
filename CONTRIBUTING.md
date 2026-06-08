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
6. Commit with clear message (see `.github/CONVENTIONAL_COMMITS.md`)
7. Push to your fork
8. Open a Pull Request

### Code Style

- Follow PEP 8 for Python code
- Use type hints where possible
- Add docstrings to functions and classes
- Run `black .` for formatting Python code
- Use ESLint for JavaScript/TypeScript code

### Testing

See [docs/TESTING.md](docs/TESTING.md) for the complete test suite reference.

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_encoder.py

# Run with coverage
pytest --cov=. tests/

# Standalone runner (subset of critical tests)
python run_tests.py

# SDK tests
cd sdk/web/universal-translation-sdk && npm install && npm test
cd sdk/react-native/UniversalTranslationSDK && npm install && npm test
cd sdk/android/UniversalTranslationSDK && ./gradlew test
cd sdk/ios/UniversalTranslationSDK && swift test
```

### Development Workflow

Use the unified CLI for all operations:

```bash
# Verify environment
./uts setup --check

# Run data pipeline (auto-resumes if interrupted)
./uts data --pipeline

# Build vocabulary
./uts vocab --build

# Train (auto-resumes, use --force to re-run)
./uts train --full --num-epochs 3

# Monitor with TUI dashboard
./uts tui --config config/base.yaml

# Evaluate
./uts eval --model

# Publish model to HF Hub
./uts publish --repo-id your-org/uts
```

### Publishing (SDKs)
- See `docs/SDK_PUBLISHING.md` for publishing Android (Maven), iOS (CocoaPods/SPM), and RN linking.
- Web SDK and PyPI packages are published via GitHub Actions (build-upload.yml, publish-pypi.yml).
- Use `uts tools --version` to check all component versions before releasing.

### Areas We Need Help
1. **Language Support**: Adding more languages and improving vocabulary packs
2. **Mobile SDKs**: Enhancing iOS/Android/Flutter/React Native implementations
3. **Optimization**: Making models smaller/faster through quantization techniques
4. **Documentation**: Improving guides and examples
5. **Testing**: Adding more test cases and integration tests
6. **TUI Dashboard**: Enhancing the Textual-based terminal UI
7. **Auto-Resume**: Improving checkpoint/resume reliability across pipeline stages

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
docker compose --env-file .env up -d
```

#### Using Role-based Install Script
```bash
# Quick development setup
bash scripts/install.sh --dev

# Full stack (training + serving + coordinator)
bash scripts/install.sh --all
```

#### Manual Setup
```bash
# Clone repo
git clone https://github.com/yourusername/universal-translation-system
cd universal-translation-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements (modular)
# Base runtime
pip install -r requirements/base.txt
# Training + serving
pip install -r requirements/train.txt -r requirements/serve.txt
# Optional service-specific extras
pip install -r requirements/decoder.txt -r requirements/coordinator.txt

# Run components individually
python cloud_decoder/optimized_decoder.py
python coordinator/advanced_coordinator.py
```

### Contributing Compute Resources

#### Usage Modes

There are two ways to run a decoder node (`udn`):

- **Private/Personal Use:**  
  You may run a decoder node for your own needs or testing without registering it with the project. No further action is needed.

- **Public Contribution:**  
  If you wish to contribute compute power to support the Universal Translation System, register your node so it can be added to the global decoder pool and serve requests for all users.

Want to help by running a decoder node? Here's how:

#### Option 1: Quick Deploy
```bash
# Install the package
pip install universal-decoder-node

# Register your node with the coordinator
udn register --name "your-node-name" --endpoint "https://your-decoder.com" --gpu-type "T4" --capacity 100 --coordinator-url "https://coordinator.example.com"

# Start the decoder service (GPU auto-detected, permission prompted)
udn start --host 0.0.0.0 --port 8001 --workers 4
```

#### Option 2: Docker (one command)
```bash
# Build & run
udn docker --gpus

# Or with Docker Compose:
docker compose --env-file .env up -d decoder
udn register --name "your-node-name" --endpoint "https://your-decoder.com"
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
- **Decoder (Cloud/Local Node)** exposes a REST API (`/decode` endpoint, served by `udn`)
- **Communication Protocol**: The encoder sends compressed embeddings (binary) to the decoder's `/decode` endpoint, specifying the target language in the header. The decoder returns the translated text as JSON.
- **Coordinator** manages the decoder pool, handles load balancing, and monitors health
- See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/API.md](docs/API.md) for protocol details.

## Our Mission

To break down language barriers by making high-quality translation accessible on every device, regardless of internet speed or hardware limitations.

We believe communication is a human right, not a luxury.

## Project Health

![Languages Supported](https://img.shields.io/badge/languages-20-brightgreen)
![Model Size](https://img.shields.io/badge/encoder%20size-35MB-blue)
![Decoder Size](https://img.shields.io/badge/decoder%20size-350MB-blue)

## Community & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/universal-translation-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/universal-translation-system/discussions)

**[Code of Conduct](CODE_OF_CONDUCT.md)**
Please be respectful and constructive in all interactions.
