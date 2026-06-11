# Universal Decoder Node

High-performance translation decoder service. Run locally via pip or Docker — your data stays on your network.

## Quick Start

```bash
pip install universal-decoder-node

# Start with GPU detection + permission prompt
udn start

# Or serve with LitServe (auto-batching, 2x faster)
udn serve
```

On first run, `udn` detects CUDA/MPS GPUs and asks permission before using them.

## Docker (one command)

```bash
udn docker              # build & run on port 8000
udn docker --gpus       # with GPU passthrough
udn docker --port 9000  # custom port
```

Or manually:

```bash
docker build -t udn .
docker run -d --gpus all -p 8000:8000 udn
```

## CLI Reference

| Command | Description |
|---|---|
| `udn start` | Start decoder (FastAPI/uvicorn) |
| `udn serve` | Start with LitServe (auto-batching) |
| `udn docker` | Build & run in Docker |
| `udn status` | Check service health |
| `udn register` | Register node with coordinator pool |
| `udn discover` | Scan network for local decoders (mDNS) |
| `udn test` | Test translation |
| `udn init` | Generate config file |

## Auto-Discovery (mDNS/Zeroconf)

When running, `udn` advertises itself on the local network via mDNS. SDKs automatically discover it:

```bash
# On any machine, discover running decoders:
udn discover
```

The Web SDK also scans `localhost:8000`, `:8080`, `:9000` automatically if no URL is configured.

## Configuration

```bash
udn init                # create config.yaml
udn start --config config.yaml
```

Key config fields:
```yaml
decoder:
  host: 0.0.0.0
  port: 8000
  device: cuda          # set by GPU permission prompt
  max_batch_size: 64
```

## GPU Support

- Auto-detects NVIDIA CUDA and Apple Metal (MPS)
- Prompts for permission on first run
- Preference saved to `decoder_config.yaml`
- Override: `udn start --gpu` / `udn start --no-gpu`

## API

| Endpoint | Description |
|---|---|
| `GET /health` | Health check |
| `POST /decode` | Decode compressed encoder output |
| `GET /metrics` | Prometheus metrics |

## Development

```bash
git clone https://github.com/code-with-zeeshan/universal-translation-system
cd universal-decoder-node
pip install -e .[dev]
pytest
black udn
isort udn
```

## License

Apache License 2.0
