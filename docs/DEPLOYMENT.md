# Deployment Guide

## Overview
This guide covers local (Docker Compose) and Kubernetes deployment for the Universal Translation System. It aligns with the current docker-compose.yml and Kubernetes manifests.

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
docker compose up -d --build encoder decoder redis coordinator prometheus grafana

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
- **Decoder**: `GET /health` (required by compose)
- **Coordinator**: `GET /api/status` (health summary)
  - Note: docker-compose.yml currently checks `GET /health` for coordinator. If using the current codebase, change the compose healthcheck to `/api/status` or add a trivial `/health` endpoint to the coordinator.

### Relevant files
- `docker-compose.yml`
- `docker/encoder.Dockerfile`, `docker/decoder.Dockerfile`, `docker/coordinator.Dockerfile`
- `monitoring/prometheus/*`, `monitoring/grafana/*`
- `configs/decoder_pool.json` (file-based pool config)

### Configuration via environment variables
See `docs/environment-variables.md` and `.env.example`.
- **Ports**: `ENCODER_PORT`, `DECODER_PORT`, `COORDINATOR_PORT`
- **Coordinator**: `POOL_CONFIG_PATH` (default: `configs/decoder_pool.json`), `REDIS_URL`
- **Secrets**: `DECODER_JWT_SECRET`, `COORDINATOR_JWT_SECRET`, `COORDINATOR_TOKEN`, `INTERNAL_SERVICE_TOKEN`

### Volumes and artifacts
- **Models**: `./models` mounted to `/app/models`
- **Vocabulary**: `./vocabulary` mounted to `/app/vocabs`
  - If you previously used `./vocabs`, either rename to `vocabulary` or update mounts consistently.
- Ensure directories exist locally before starting containers.

### GPU notes
- The `decoder` service requests one NVIDIA GPU and sets `CUDA_VISIBLE_DEVICES=0`.
- Install NVIDIA Container Toolkit and confirm `--gpus all` support: `docker run --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi`.

### Preflight check (recommended)
```bash
python -m tools.cloud_preflight
```
Exit code 0 indicates you’re good to proceed.

## 2) Decoder only (Standalone Docker)
```bash
# Build
docker build -f docker/decoder.Dockerfile -t universal-decoder:latest .

# Run (GPU required)
docker run --gpus all -p 8001:8001 \
  -e DECODER_JWT_SECRET=replace-with-strong-secret \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/vocabulary:/app/vocabs" \
  universal-decoder:latest
```

## 3) Kubernetes Deployment
Manifests live in `kubernetes/`.
```bash
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/encoder-artifacts-pvc.yaml
kubectl apply -f kubernetes/encoder-build.yaml
kubectl apply -f kubernetes/decoder-deployment.yaml
kubectl apply -f kubernetes/decoder-service.yaml
```
Notes:
- **GPU resources**: Configure node selectors/taints and NVIDIA device plugin on nodes.
- **Secrets/Configs**: Use Kubernetes Secrets for tokens/keys; ConfigMaps for non-secret configs.
- **Coordinator + Redis**: Create corresponding Deployment/Service objects (compose shows env expectations).

## 4) Monitoring and scaling
- **Health**: `/health` (decoder), `/api/status` (coordinator)
- **Metrics**: `/metrics` on decoder and coordinator (Prometheus scrapes)
- **Horizontal scaling**: run multiple `decoder` replicas; Coordinator routes to least-loaded
- **Vertical scaling**: larger GPUs; tune batch size and workers

## 5) Security
- **Secrets**: Set strong values in `.env` (JWTs, tokens) and in K8s Secrets
- **Transport**: Terminate TLS at ingress/proxy
- **Access**: Restrict coordinator admin endpoints; require `COORDINATOR_TOKEN`/JWT
- **Abuse protection**: Enable rate limiting; validate inputs

### Secret rotation rollout (Docker Compose)
1. Generate new secrets/keys locally using the rotation CLI:
   ```bash
   python tools/rotate_secrets.py --type hs256 --key coordinator_jwt_secret --set-env
   python tools/rotate_secrets.py --type hs256 --key decoder_jwt_secret --set-env
   python tools/rotate_secrets.py --type rs256 --kid roll-$(date +%Y%m%d) --set-env
   ```
2. Update `.env` or secret files (preferred: mount as `*_FILE`):
   - Write new HS256 secrets to files and reference via `*_FILE`.
   - For RS256, add the new public key to `JWT_PUBLIC_KEY` (use `||`-separated PEMs) and keep the old key for a grace window.
3. Deploy with both old and new keys present (grace period):
   - Coordinator/Decoder will build JWKS from the combined set and accept both kids.
4. Redeploy services:
   ```bash
   docker compose up -d --build coordinator decoder
   ```
5. After clients are updated and tokens issued with the new key only, remove the old key from env/files and redeploy again.

### Secret rotation rollout (Kubernetes)
1. Create new secrets (HS256 and/or RS256):
   ```bash
   # Example: HS256
   kubectl create secret generic uts-auth-secrets-new \
     --from-literal=COORDINATOR_JWT_SECRET=... \
     --from-literal=DECODER_JWT_SECRET=... 

   # Example: RS256 public(s) and private
   kubectl create secret generic uts-rs256-new \
     --from-literal=JWT_PUBLIC_KEY="$(cat pub_new.pem)" \
     --from-literal=JWT_PRIVATE_KEY="$(cat priv_new.pem)"
   ```
2. Patch Deployments to mount new secrets alongside existing ones:
   - Use parallel env vars or preferred: mount to files and reference via `*_FILE`.
3. Ensure both old and new keys are live for a grace window; JWKS will include both and decoder/coordinator will accept both kids.
4. Roll out with zero downtime:
   ```bash
   kubectl rollout restart deployment/decoder
   kubectl rollout restart deployment/coordinator
   kubectl rollout status deployment/decoder
   kubectl rollout status deployment/coordinator
   ```
5. After migration, remove old keys from Secrets and update Deployments.

See also: `.github/workflows/scheduled-rotation.yml` for an automated rotation example and `docs/SECURITY_BEST_PRACTICES.md` for policy details.

## 6) Migration tips and known mismatches
- **Coordinator port**: Compose sets container API port to `8002` (older docs referenced `5100`).
- **Vocabulary path**: Use `./vocabulary` locally and mount to `/app/vocabs` in containers.
- **Decoder model filename**: Default in code expects `models/production/decoder.pt`. Update your model export or set envs/configs accordingly.
- **Coordinator healthcheck**: Update compose healthcheck to `/api/status` or add `/health` to coordinator.

## 7) Artifacts expected at runtime
- **Models** (mounted to `/app/models`):
  - `models/production/decoder.pt` (default expected by decoder)
  - `models/model_registry.json` (optional registry)
- **Vocabulary packs** (mounted to `/app/vocabs`):
  - Files created by vocab tooling; a `manifest.json` if produced
- **Checkpoints/logs** (optional for runtime): under `checkpoints/` and `logs/`

### How services use them
- **Decoder**: loads weights from `/app/models/production/decoder.pt`; loads vocabulary packs from `/app/vocabs` based on requested languages.
- **Coordinator**: routes to least-loaded healthy decoder using configured pool (file or discovery).
- **SDKs**: encode on-device where supported; call coordinator `/api/decode` (binary) for decoding, or decoder directly in single-node setups.