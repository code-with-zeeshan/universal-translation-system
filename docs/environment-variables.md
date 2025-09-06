# Environment Variables

This document lists the key environment variables used across services and SDKs. Use a `.env` file, shell exports, Docker Compose, or K8s Secrets/ConfigMaps to set these.

## General
- **MODEL_VERSION**: Model version string (default: 1.0.0)
- **LOG_LEVEL**: Set logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO.

## Decoder Node (cloud_decoder)
- **API_HOST**: Bind host (default: 0.0.0.0)
- **API_PORT**: Port (default: 8001)
- **API_WORKERS**: Workers (default: 1)
- **API_TITLE**: Title (default: Cloud Decoder API)
- **DECODER_JWT_SECRET**: JWT secret for admin endpoints (CRITICAL)
- **DECODER_JWT_SECRET_FILE**: Alternative to DECODER_JWT_SECRET; path to file containing the secret (recommended in Docker/K8s) (CRITICAL)
- **DECODER_CONFIG_PATH**: Path to YAML config (default: config/decoder_config.yaml)
- **HF_HUB_REPO_ID**: HF repo for adapters/models (default: your-hf-org/universal-translation-system)
- **CUDA_VISIBLE_DEVICES**: GPU selection (e.g., 0)
- **OMP_NUM_THREADS**: Inference CPU threads
- **PREFETCH_VOCAB_GROUPS**: Comma‑separated vocab packs to prefetch (optional)
- **PREFETCH_ADAPTERS**: Comma‑separated adapters to prefetch (optional)

RS256 (optional):
- **JWT_PRIVATE_KEY_FILE**: Path to RS256 private key (PEM)
- **JWT_PUBLIC_KEY_PATH**: Path(s) to RS256 public key(s). Supports `||`-separated list for rotation.

## Coordinator
- **API_HOST**: Bind host (default: 0.0.0.0)
- **API_PORT**: Port (default: 8002 under Compose)
- **API_WORKERS**: Workers (default: 1)
- **API_TITLE**: Title (default: Universal Translation Coordinator)
- (CRITICAL) **COORDINATOR_SECRET** or **COORDINATOR_SECRET_FILE**: Cookie/session secret
- (CRITICAL) **COORDINATOR_JWT_SECRET** or **COORDINATOR_JWT_SECRET_FILE**: JWT secret for admin APIs
- (CRITICAL) **COORDINATOR_TOKEN** or **COORDINATOR_TOKEN_FILE**: Admin login token (dashboard)
- (CRITICAL) **INTERNAL_SERVICE_TOKEN** or **INTERNAL_SERVICE_TOKEN_FILE**: Token for internal calls to decoder (e.g., compose_adapter)
- **POOL_CONFIG_PATH**: File path for decoder pool (default: configs/decoder_pool.json)
- **REDIS_URL**: Redis connection URL (optional, enables Redis features)
- **COORDINATOR_MIRROR_INTERVAL**: Seconds between Redis→disk mirrors of decoder pool (default: 60; minimum enforced: 5; effective value logged at startup)
- **ETCD_HOST**: etcd host (optional)
- **ETCD_PORT**: etcd port (optional)
- **USE_ETCD**: enable etcd service discovery (true/false)
- **SERVICE_TTL**: service discovery TTL seconds

RS256 (optional):
- **JWT_PRIVATE_KEY_FILE**: Path to RS256 private key (PEM)
- **JWT_PUBLIC_KEY_PATH**: Path(s) to RS256 public key(s). Supports `||`-separated list for rotation.

## Docker Compose Ports
- **ENCODER_PORT**: 8000 by default
- **DECODER_PORT**: 8001 by default
- **COORDINATOR_PORT**: 8002 by default
- **PROMETHEUS_PORT**: 9090 by default
- **GRAFANA_PORT**: 3000 by default

## SDKs
- Web SDK:
  - **DECODER_API_URL**: Point to coordinator `/api/decode` or decoder `/decode`
  - **WASM_ENCODER_PATH**: Path to WASM artifacts (if using WASM)
  - Optional toggles: `USE_WASM_ENCODER`, `ENABLE_FALLBACK`
- React Native SDK:
  - **USE_NATIVE_ENCODER**: true/false

## Vocabulary/Training (commonly used)
- **ENCODER_MODEL_PATH**: models/production/encoder.pt
- **FALLBACK_MODEL_PATH**: models/fallback/encoder.pt
- **EMBEDDING_DIM**: 768 (example)

## Monitoring
- **METRICS_PATH**: /metrics
- **METRICS_COLLECTION_INTERVAL**: seconds (default: 15)
- **ENABLE_SYSTEM_METRICS**: true/false
- **ENABLE_GPU_METRICS**: true/false
- **ENABLE_VOCABULARY_METRICS**: true/false

## Security Tips
1. Replace all defaults in production; store secrets in a vault/Secrets.
2. Terminate TLS at your ingress/proxy.
3. Rotate JWT secrets periodically.

## Troubleshooting
- Verify envs are set inside containers: `docker compose exec <svc> env | sort`
- In K8s, check `kubectl describe deploy <name>` and ConfigMaps/Secrets mounting.
- See TROUBLESHOOT.md for more.