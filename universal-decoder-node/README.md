# Universal Decoder Node

High-performance translation decoder service for the Universal Translation System. Serves as a standalone node or registers with the coordinator pool.

## Features
- GPU-accelerated decoding with continuous batching
- Built-in monitoring with Prometheus metrics
- JWT authentication for admin endpoints
- Docker support with GPU passthrough
- Optimized for T4/V100/A100 GPUs
- RESTful API with FastAPI/uvicorn

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
- `start` -- Start the decoder service
- `status` -- Check service status
- `register` -- Register node with coordinator
- `test` -- Test translation
- `init` -- Create configuration file
- `docker-build` -- Build Docker image

## Configuration

```bash
universal-decoder-node init
```

Example `config.yaml`:
```yaml
decoder:
  host: 0.0.0.0
  port: 8001
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
```bash
universal-decoder-node docker-build
docker run -d \
  --gpus all \
  -p 8001:8001 \
  -v ./models:/app/models \
   -v ./vocabulary/vocab:/app/vocabs \
  universal-decoder:latest
```

## API Endpoints
- `GET /health` -- Health check
- `GET /status` -- Service status
- `POST /decode` -- Decode translation
- `POST /admin/reload_model` -- Reload model (requires auth)

## Development
```bash
git clone https://github.com/yourusername/universal-decoder-node
cd universal-decoder-node
pip install -e .[dev]
pytest
black universal_decoder_node
isort universal_decoder_node
```

## License
Apache License 2.0
