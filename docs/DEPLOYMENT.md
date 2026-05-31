# Deployment Guide

## Overview
This guide covers local (Docker Compose), standalone Docker, Helm, and Kubernetes deployment for the Universal Translation System.

## 1) Local Stack (Docker Compose)

### Prerequisites
- **Docker** and **Docker Compose**
- **NVIDIA drivers + CUDA + NVIDIA Container Toolkit** (for GPU decoder)
- **.env** (optional) based on `.env.example`

### Bring up the stack
```bash
# From repo root
# Optional: create .env and set secrets (JWTs, tokens) before running

# Build and start encoder, decoder, redis, coordinator, prometheus, grafana
docker compose --env-file .env up -d --build encoder decoder redis coordinator prometheus grafana

# View logs for a service
docker compose logs -f decoder
```

### Default ports
- **Encoder**: http://localhost:8000
- **Decoder**: http://localhost:8001
- **Coordinator**: http://localhost:8002
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

### Health endpoints
- **Decoder**: `GET /health`
- **Coordinator**: `GET /api/status` (health summary)

### Relevant files
- `docker-compose.yml` (root, full dev stack)
- `docker/docker-compose.yml` (production-oriented with Docker secrets)
- `docker/encoder.Dockerfile`, `docker/decoder.Dockerfile`, `docker/coordinator.Dockerfile`, `docker/Dockerfile.cloud`
- `monitoring/prometheus/*`, `monitoring/grafana/*`

### Configuration via environment variables
See `docs/environment-variables.md` and `.env.example`.
- **Ports**: `ENCODER_PORT`, `DECODER_PORT`, `COORDINATOR_PORT`
- **Coordinator**: `POOL_CONFIG_PATH` (default: `configs/decoder_pool.json`), `REDIS_URL`
- **Secrets**: `DECODER_JWT_SECRET`, `COORDINATOR_JWT_SECRET`, `COORDINATOR_TOKEN`, `INTERNAL_SERVICE_TOKEN`

### Volumes and artifacts
- **Models**: `./models` mounted to `/app/models`
  - Expected: `/app/models/production/decoder.pt`
  - Optional: `/app/models/model_registry.json`
- **Vocabulary**: `./vocabs` mounted to `/app/vocabs`
  - Contents: vocabulary packs; optional `manifest.json`
- Ensure directories exist locally before starting containers.

See also: `docs/ONBOARDING.md` and `docs/Vocabulary_Guide.md`.

### GPU notes
- The `decoder` service requests one NVIDIA GPU and sets `CUDA_VISIBLE_DEVICES=0`.
- Install NVIDIA Container Toolkit and confirm: `docker run --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi`.

### Preflight check (recommended)
```bash
python -m tools.cloud_preflight
```

## 2) Decoder only (Standalone Docker)
```bash
# Build
docker build -f docker/decoder.Dockerfile -t universal-decoder:latest .

# Run (GPU required)
docker run --gpus all -p 8001:8001 \
  -e DECODER_JWT_SECRET=replace-with-strong-secret \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/vocabs:/app/vocabs" \
  universal-decoder:latest
```

## 3) Helm Chart Deployment (Recommended for K8s)

```bash
# Deploy all services via Helm
helm upgrade --install uts ./charts/uts \
  --set secrets.decoderJwtSecret=<base64> \
  --set secrets.coordinatorSecret=<base64> \
  --set secrets.coordinatorJwtSecret=<base64> \
  --set secrets.coordinatorToken=<base64> \
  --set secrets.internalServiceToken=<base64>
```

The Helm chart configures:
- Coordinator service (port 5100)
- Decoder service (port 8001)
- Encoder service (port 8000)
- Redis with healthchecks
- Secrets via `secrets.yaml` template
- Configurable resource requests/limits

## 4) Kubernetes Deployment (Manifest-based)

Manifests live in `kubernetes/`.
```bash
# Apply secrets first
kubectl apply -f kubernetes/secrets.yaml

# Then deployments
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/encoder-artifacts-pvc.yaml
kubectl apply -f kubernetes/encoder-build.yaml
kubectl apply -f kubernetes/decoder-deployment.yaml
kubectl apply -f kubernetes/decoder-service.yaml
kubectl apply -f kubernetes/coordinator-deployment.yaml
kubectl apply -f kubernetes/coordinator-service.yaml
```

Notes:
- **GPU resources**: Configure node selectors/taints and NVIDIA device plugin.
- **Secrets/Configs**: Use Kubernetes Secrets for tokens/keys.
- **Coordinator + Redis**: Redis service is auto-configured when `REDIS_URL` is set.

## 5) Monitoring and scaling
- **Health**: `/health` (decoder), `/api/status` (coordinator)
- **Metrics**: `/metrics` on decoder, coordinator, and encoder
- **Horizontal scaling**: Multiple decoder replicas; Coordinator routes to least-loaded
- **Autoscaling**: HPAs in `charts/uts/` for CPU-based scaling

## 6) Security
- **Secrets**: Set strong values in `.env` (JWTs, tokens) and K8s Secrets
- **Transport**: Terminate TLS at ingress/proxy
- **Access**: Restrict coordinator admin endpoints; require `COORDINATOR_TOKEN`/JWT
- **Abuse protection**: Enable rate limiting; validate inputs
- **Container security**: All Dockerfiles use non-root users

### Secret rotation rollout (Docker Compose)
1. Generate new secrets/keys:
   ```bash
   python tools/rotate_secrets.py --type hs256 --key coordinator_jwt_secret --set-env
   python tools/rotate_secrets.py --type hs256 --key decoder_jwt_secret --set-env
   python tools/rotate_secrets.py --type rs256 --kid roll-$(date +%Y%m%d) --set-env
   ```
2. Update `.env` or secret files (prefer `*_FILE` mount).
3. Redeploy: `docker compose up -d --build coordinator decoder`

### Secret rotation rollout (Kubernetes)
1. Create new secrets.
2. Patch Deployments to mount new secrets alongside existing ones.
3. Ensure both old and new keys are live for a grace window.
4. `kubectl rollout restart deployment/decoder && kubectl rollout status deployment/decoder`

See also: `docs/SECURITY_BEST_PRACTICES.md`.

## 7) Artifacts expected at runtime
- **Models** (mounted to `/app/models`):
  - `models/production/decoder.pt` (default expected by decoder)
  - `models/production/encoder.pt` (default expected by encoder)
  - `models/model_registry.json` (optional)
- **Vocabulary packs** (mounted to `/app/vocabs`):
  - `.msgpack` files created by vocab tooling
- **Checkpoints/logs** (optional): under `checkpoints/` and `logs/`

### How services use them
- **Decoder**: loads weights from `/app/models/production/decoder.pt`; vocabulary packs from `/app/vocabs`.
- **Coordinator**: routes to least-loaded healthy decoder using configured pool (file or Redis).
- **SDKs**: encode on-device; call coordinator `/api/decode` or decoder `/decode`.
