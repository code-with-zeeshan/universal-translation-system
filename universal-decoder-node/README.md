# universal-decoder-node/README.md
# Universal Decoder Node

High-performance translation decoder service for the Universal Translation System.

## Features

- 🚀 GPU-accelerated decoding with continuous batching
- 📊 Built-in monitoring with Prometheus metrics
- 🔒 JWT authentication for admin endpoints
- 🐋 Docker support with GPU passthrough
- ⚡ Optimized for T4/V100/A100 GPUs
- 🌐 RESTful API with FastAPI

## Installation

```bash
pip install universal-decoder-node
```

For GPU support:
```bash
pip install universal-decoder-node[gpu]
```

## Quick Start

1. Start the decoder service:
```bash
universal-decoder-node start
```

2. Check service health:
```bash
universal-decoder-node status
```

3. Register with coordinator:
```bash
universal-decoder-node register \
  --name my-decoder \
  --endpoint https://my-decoder.com \
  --gpu-type T4
```

## CLI Commands

- `start` - Start the decoder service
- `status` - Check service status
- `register` - Register node with coordinator
- `test` - Test translation
- `init` - Create configuration file
- `docker-build` - Build Docker image

## Configuration

Create a configuration file:
```bash
universal-decoder-node init
```

Example `config.yaml`:
```yaml
decoder:
  host: 0.0.0.0
  port: 8000
  workers: 1
  model_path: models/decoder.pt
  vocab_dir: vocabs
  device: cuda
  max_batch_size: 64
  batch_timeout_ms: 10

security:
  jwt_secret: your-secret-key
  enable_auth: true

monitoring:
  prometheus_port: 9200
  enable_tracing: false
```

## Docker Deployment

Build and run with Docker:
```bash
universal-decoder-node docker-build
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -v ./models:/app/models \
  -v ./vocabs:/app/vocabs \
  universal-decoder:latest
```

## API Endpoints

- `GET /health` - Health check
- `GET /status` - Service status
- `POST /decode` - Decode translation
- `POST /admin/reload_model` - Reload model (requires auth)

## Development

```bash
# Clone the repository
git clone https://github.com/yourusername/universal-decoder-node
cd universal-decoder-node

# Install in development mode
pip install -e .[dev]

# Run tests
pytest

# Format code
black universal_decoder_node
isort universal_decoder_node
```
```
## License

Apache License 2.0
```

## Usage Examples

After installation:

```bash
# 1. Start decoder with default settings
universal-decoder-node start

# 2. Start with custom configuration
universal-decoder-node start --config my-config.yaml

# 3. Start with specific model
universal-decoder-node start --model-path models/my-decoder.pt --vocab-dir my-vocabs

# 4. Run in Docker
universal-decoder-node start --docker

# 5. Check status
universal-decoder-node status --detailed

# 6. Register with coordinator
universal-decoder-node register \
  --name gpu-node-1 \
  --endpoint https://my-decoder.example.com \
  --region us-west-2 \
  --gpu-type A100 \
  --capacity 200

# 7. Test translation
universal-decoder-node test \
  --text "Hello world" \
  --source-lang en \
  --target-lang es
```

This package structure provides:

1. **Core decoder functionality** 
2. **CLI interface** 
3. **Docker support** for easy deployment
4. **Configuration management**
5. **Monitoring and health checks**
6. **Registration with coordinator**
7. **Testing utilities**