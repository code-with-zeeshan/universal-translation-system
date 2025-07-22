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
- Run `black .` for formatting

### Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_encoder.py

# Run with coverage
pytest --cov=. tests/
```

### Areas We Need Help
1. **Language Support**: Adding more languages
2. **Mobile SDKs**: Improving iOS/Android implementations
3. **Optimization**: Making models smaller/faster
4. **Documentation**: Improving guides and examples
5. **Testing**: Adding more test cases

### Development Setup
```bash
# Clone repo
git clone https://github.com/code-with-zeeshan/universal-translation-system
cd universal-translation-system

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
# Or install in development mode
pip install -e .

# Run tests
pytest
```

### Contributing Compute Resources

Want to help by running a decoder node? Here's how:

#### Option 1: Quick Deploy (Managed)
```bash
pip install universal-decoder-node
universal-decoder-node register --name "my-node" --gpu "T4"
universal-decoder-node start
```

#### Option 2: Custom Deploy
1. Deploy the decoder to your cloud provider or on-prem server (see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md))
2. Ensure it meets minimum requirements (T4 GPU, 16GB RAM)
3. Register your node by submitting a PR to `configs/decoder_pool.json`:

```json
{
  "node_id": "your-unique-id",
  "endpoint": "https://your-decoder.com",
  "region": "us-east-1",
  "gpu_type": "T4",
  "capacity": 100
}
```

**Decoder Node Requirements**
- HTTPS endpoint with valid certificate
- 99% uptime commitment
- Response time < 100ms (p95)
- Support for health checks (`/health` endpoint)

**Benefits for Contributors**
- Recognition in project contributors list
- Usage statistics dashboard access
- Priority support for issues

### How Encoder and Decoder Communicate
- **Encoder (Edge/SDK)** encodes text to embeddings on device
- **Decoder (Cloud Node)** exposes a REST API (e.g., `/decode` endpoint, now served by Litserve)
- **Communication Protocol**: The encoder sends compressed embeddings (binary) to the decoder's `/decode` endpoint, specifying the target language in the header. The decoder returns the translated text as JSON.
- See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/API.md](docs/API.md) for protocol details.

## 🌍 Our Mission

To break down language barriers by making high-quality translation accessible on every device, regardless of internet speed or hardware limitations.

We believe communication is a human right, not a luxury

## 📊 Project Health

![Downloads](https://img.shields.io/npm/dt/universal-translation-sdk)
![Contributors](https://img.shields.io/github/contributors/code-with-zeeshan/universal-translation-system)
![Languages Supported](https://img.shields.io/badge/languages-20-brightgreen)
![Model Size](https://img.shields.io/badge/encoder%20size-35MB-blue)
![Decoder Size](https://img.shields.io/badge/decoder%20size-350MB-blue)

## Community & Support

- **Issues**: [GitHub Issues](https://github.com/code-with-zeeshan/universal-translation-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/code-with-zeeshan/universal-translation-system/discussions)
- **Email**: opensource@yourdomain.com (checked weekly)

> **Note**: This is a solo developer project. Response times may vary. For urgent issues, please use GitHub Issues.

**[Code of Conduct](CODE_OF_CONDUCT.md)**
Please be respectful and constructive in all interactions.