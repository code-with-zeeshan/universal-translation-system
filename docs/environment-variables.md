# Environment Variables

Complete reference for all environment variables used across services, SDKs, and tools.
Set via `.env` file, shell exports, Docker Compose, or K8s Secrets/ConfigMaps.

## Secrets & Credentials

| Variable | Required | Default | Description |
|---|---|---|---|
| `UTS_HMAC_KEY` | **YES** | — | HMAC key for secure serialization (≥32 chars). Generate: `openssl rand -hex 32` |
| `UTS_ROLE` | No | `general` | Role for secret bootstrap: `general`, `coordinator`, `decoder` |
| `UTS_JWT_DEFAULT_ALG` | No | `HS256` | Default JWT algorithm: `HS256` or `RS256` |
| `JWT_SECRET` | No | — | Fallback JWT secret (use named vars below instead) |
| `COORDINATOR_SECRET` | **YES** | — | Cookie/session secret for coordinator dashboard |
| `COORDINATOR_SECRET_FILE` | No | — | Path to file containing `COORDINATOR_SECRET` (preferred in Docker/K8s) |
| `COORDINATOR_JWT_SECRET` | **YES** | — | JWT secret for coordinator admin APIs |
| `COORDINATOR_JWT_SECRET_FILE` | No | — | Path to file containing `COORDINATOR_JWT_SECRET` |
| `COORDINATOR_TOKEN` | **YES** | — | Admin login token for coordinator dashboard |
| `COORDINATOR_TOKEN_FILE` | No | — | Path to file containing `COORDINATOR_TOKEN` |
| `INTERNAL_SERVICE_TOKEN` | **YES** | — | Token for internal coordinator→decoder calls |
| `INTERNAL_SERVICE_TOKEN_FILE` | No | — | Path to file containing `INTERNAL_SERVICE_TOKEN` |
| `DECODER_JWT_SECRET` | **YES** | — | JWT secret for decoder admin endpoints |
| `DECODER_JWT_SECRET_FILE` | No | — | Path to file containing `DECODER_JWT_SECRET` |
| `JWT_PRIVATE_KEY` | No | — | RS256 private key (PEM) — inline (use `*_FILE` variant in production) |
| `JWT_PRIVATE_KEY_FILE` | No | — | Path to RS256 private key (PEM) |
| `JWT_PUBLIC_KEY` | No | — | RS256 public key(s) (PEM). Supports `\|\|`-separated list for key rotation |
| `JWT_PUBLIC_KEY_PATH` | No | — | Path(s) to RS256 public key(s). Supports `\|\|`-separated list |
| `JWT_KEY_IDS` | No | — | `\|\|`-separated list of active key IDs for RS256 |
| `JWT_ISS` | No | — | JWT issuer claim |
| `JWT_AUD` | No | — | JWT audience claim |
| `JWKS_KEYS_JSON` | No | — | JWKS key set as JSON array (for public key distribution) |
| `UTS_VOCAB_SIGNING_KEY` | No | — | HMAC key for vocabulary pack signing/validation |
| `REDIS_PASSWORD` | Yes* | — | Redis password (required if `REDIS_URL` not set) |
| `HF_TOKEN` | **YES** | — | Hugging Face token for model/vocab downloads |
| `GRAFANA_ADMIN_PASSWORD` | No | `admin` | Grafana admin password |

### Secret Bootstrap Resolution Order

The `*_FILE` pattern is the recommended approach for containers:
1. If `VAR_FILE` is set and points to a readable file, `VAR` is loaded from that file
2. Otherwise `VAR` is read directly from environment
3. Both fall back to credential manager (`CredentialManager` with keyring + encrypted file)

## Services: Host / Port / Workers

| Variable | Default | Description |
|---|---|---|
| `DECODER_HOST` | `0.0.0.0` | Decoder bind host |
| `DECODER_PORT` | `8001` | Decoder port |
| `DECODER_WORKERS` | `1` | Decoder uvicorn workers |
| `DECODER_API_URL` | — | External decoder API URL (SDK-facing) |
| `DECODER_CONFIG_PATH` | `config/decoder_config.yaml` | Decoder YAML config path |
| `DECODER_CONFIG` | `decoder_config.yaml` | Alternative config path (used by `udn` CLI) |
| `DECODER_ENDPOINT` | `http://localhost:8001` | Decoder endpoint URL |
| `DECODER_API_VERSION` | — | API version override |
| `ENCODER_PORT` | `8000` | Encoder port |
| `ENCODER_API_URL` | — | External encoder API URL (SDK-facing) |
| `API_HOST` | `0.0.0.0` | FastAPI bind host (shared by decoder & coordinator) |
| `API_PORT` | `8001` | FastAPI port |
| `API_WORKERS` | `1` | FastAPI workers |
| `API_TITLE` | `Cloud Decoder API` | FastAPI title |
| `COORDINATOR_HOST` | `0.0.0.0` | Coordinator bind host |
| `COORDINATOR_PORT` | `5100` | Coordinator port |
| `COORDINATOR_WORKERS` | `1` | Coordinator workers |
| `COORDINATOR_TITLE` | `Universal Translation Coordinator` | Coordinator title |
| `COORDINATOR_URL` | `http://localhost:5100` | Coordinator URL |
| `COORDINATOR_API_KEY` | — | API key for decoder node registration |
| `COORDINATOR_MIRROR_INTERVAL` | `30` | Seconds between Redis→disk mirrors (min: 5) |
| `PROMETHEUS_PORT` | `9090` | Prometheus port |
| `GRAFANA_PORT` | `3000` | Grafana port |

## Paths

> **Deprecated in favor of `RuntimeDirectoryManager`** (`utils.common_utils.RuntimeDirectoryManager`). These env vars still work for backward compatibility (via `utils/constants.py`), but new code should obtain runtime paths through RDM. The env vars override RDM defaults when set.

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `/app/models/decoder_model.pt` | Decoder model path (universal-decoder-node) |
| `MODEL_VERSION` | `1.0.0` | Model version string |
| `MODELS_DIR` | `models` | Models directory |
| `ENCODER_MODEL_PATH` | `models/production/encoder.pt` | Encoder model file path |
| `DECODER_MODEL_PATH` | `models/production/decoder.pt` | Decoder model file path |
| `FALLBACK_MODEL_PATH` | `models/fallback/encoder.pt` | Fallback encoder path |
| `VOCAB_DIR` | `vocabulary/vocab` | Vocabulary directory |
| `VOCABS_DIR` | `vocabulary/vocab` | Vocabulary directory (HF artifact store) |
| `ADAPTERS_DIR` | `models/adapters` | Adapter checkpoints directory |
| `CHECKPOINT_DIR` | `checkpoints` | Training checkpoints directory |
| `POOL_CONFIG_PATH` | `config/decoder_pool.json` | Decoder pool config file |
| `DATA_RAW_DIR` | `data/raw` | Raw data directory |
| `DATA_PROCESSED_DIR` | `data/processed` | Processed data directory |
| `DATA_SAMPLED_DIR` | `data/sampled` | Sampled data directory |
| `DATA_FINAL_DIR` | `data/final` | Final training-ready data |
| `DATA_CACHE_DIR` | `data/cache` | Data cache directory |
| `DATA_ESSENTIAL_DIR` | `data/essential` | Essential data directory |
| `LOG_DIR` | `logs` | Log files directory |
| `DEFAULT_CONFIG_PATH` | `config/default_config.json` | Default config path |
| `DEFAULT_MODEL_PATH` | `models/default_model` | Default model path |
| `DEFAULT_VOCAB_PATH` | `vocabulary/default_vocab` | Default vocab path |
| `DEFAULT_LOG_PATH` | `logs/system.log` | Default log file |
| `TRUSTED_MODELS_DIR` | `models` | Trusted models directory |
| `TRUSTED_HASHES_FILE` | `trusted_model_hashes.txt` | File with trusted model hashes |

### Path Overrides via `UTS_` Prefix

All paths in `utils/constants.py` accept both `VAR` and `UTS_VAR` forms:

| Variable | Default | Description |
|---|---|---|
| `UTS_MODELS_DIR` | `models` | Top-level models directory |
| `UTS_MODELS_PRODUCTION_DIR` | `models/production` | Production model artifacts |
| `UTS_MODELS_ENCODER_DIR` | `models/encoder` | Encoder-specific models |
| `UTS_MODELS_DECODER_DIR` | `models/decoder` | Decoder-specific models |
| `UTS_MODELS_ADAPTERS_DIR` | `models/adapters` | Adapter checkpoints |
| `UTS_CHECKPOINT_DIR` | `checkpoints` | Training checkpoints |
| `UTS_VOCAB_DIR` | `vocabulary/vocab` | Vocabulary packs |
| `UTS_CONFIG_DIR` | `config` | Configuration files |
| `UTS_LOG_DIR` | `logs` | Log output |
| `UTS_DATA_RAW_DIR` | `data/raw` | Raw downloads |
| `UTS_DATA_PROCESSED_DIR` | `data/processed` | Processed data |
| `UTS_DATA_SAMPLED_DIR` | `data/sampled` | Sampled data |
| `UTS_DATA_FINAL_DIR` | `data/final` | Final training data |
| `UTS_DATA_CACHE_DIR` | `data/cache` | Data cache |
| `UTS_DATA_ESSENTIAL_DIR` | `data/essential` | Essential data |

### SDK Web Paths

| Variable | Default | Description |
|---|---|---|
| `MODEL_URL` | `/models/universal_encoder.onnx` | Model URL path (Web SDK) |
| `VOCAB_URL` | `/vocabs` | Vocab URL path (Web SDK) |
| `WASM_ENCODER_PATH` | `/wasm/encoder.js` | WASM encoder path (Web SDK) |

## Feature Flags

| Variable | Default | Description |
|---|---|---|
| `ENABLE_SYSTEM_METRICS` | `true` | System-level Prometheus metrics |
| `ENABLE_GPU_METRICS` | `true` | GPU utilization metrics |
| `ENABLE_VOCABULARY_METRICS` | `true` | Vocabulary cache/pack metrics |
| `ENABLE_PROFILING` | `false` | Enable profiling mode |
| `ENABLE_MEMORY_MONITORING` | `true` | Memory usage monitoring |
| `ENABLE_FALLBACK` | `true` | Enable fallback model on decode failure |
| `AUTO_MEMORY_CLEANUP` | `true` | Automatic GPU memory cleanup |
| `USE_WASM_ENCODER` | `true` | Use WASM encoder in Web SDK |
| `USE_NATIVE_ENCODER` | `true` | Use native encoder in RN SDK |
| `USE_ETCD` | `false` | Enable etcd service discovery |
| `ENFORCE_HTTPS` | `false` | Enforce HTTPS redirects |
| `REDIS_USE_MSGPACK` | `true` | Use MessagePack for Redis payloads |
| `REDIS_HEALTH_CHECK_ENABLED` | `true` | Redis health check pings |
| `UTS_API_KEYS_USE_CREDMGR` | `false` | Store API key metadata via CredentialManager |
| `UTS_TEST_MODE` | — | Set to `1` to skip external deps in tests |

## Inference / Model Hyperparameters

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_DIM` | `512` | Embedding dimension (unified encoder/decoder) |
| `MAX_SEQUENCE_LENGTH` | `512` | Max input sequence length |
| `ENCODER_DEVICE` | `auto` | Encoder device (`cpu`, `cuda`, `mps`, `auto`) |
| `DECODER_DEVICE` | `auto` | Decoder device (`cpu`, `cuda`, `auto`) |
| `DECODER_BATCH_SIZE` | `32` | Decoder batch size |
| `DECODER_BEAM_SIZE` | `5` | Beam search width |
| `MAX_BATCH_TOKENS` | `8192` | Max tokens per batch |
| `MAX_BATCH_SIZE` | `64` | Max batch size (K8s deployment) |
| `BATCH_TIMEOUT_MS` | `10` | Batch accumulation timeout |
| `TOKEN_EXPIRY` | `3600` | JWT token expiry in seconds |

## Redis

| Variable | Default | Description |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `REDIS_PASSWORD` | — | Redis password |
| `REDIS_KEY_PREFIX` | `translation:` | Redis key prefix |
| `REDIS_CONN_TIMEOUT` | `2` | Connection timeout (seconds) |
| `REDIS_READ_TIMEOUT` | `2` | Read timeout (seconds) |
| `REDIS_HEALTH_CHECK_INTERVAL` | `30` | Health check interval (seconds) |

## Performance / Monitoring

| Variable | Default | Description |
|---|---|---|
| `MONITORING_LOG_LEVEL` | `INFO` | Log level for monitoring |
| `LOG_LEVEL` | `INFO` | Application log level |
| `LOG_FORMAT` | — | Custom log format string |
| `PROFILE_OUTPUT_DIR` | `profiles` | Profiling output directory |
| `PROFILE_EXPORT_FORMAT` | `json` | Profile export format |
| `BOTTLENECK_THRESHOLD_MS` | `100.0` | Bottleneck detection threshold (ms) |
| `MEMORY_MONITOR_INTERVAL_SECONDS` | `60` | Memory check interval |
| `MEMORY_THRESHOLD_PERCENT` | `85` | Memory warning threshold (%) |
| `GPU_MEMORY_THRESHOLD_PERCENT` | `85` | GPU memory warning threshold (%) |
| `CLEANUP_THRESHOLD_PERCENT` | `80` | Auto-cleanup trigger (%) |
| `METRICS_PATH` | `/metrics` | Prometheus metrics endpoint path |
| `METRICS_COLLECTION_INTERVAL` | `15` | Metrics collection interval (seconds) |

## Circuit Breaker

| Variable | Default | Description |
|---|---|---|
| `CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `5` | Failures before circuit opens |
| `CIRCUIT_BREAKER_RECOVERY_TIMEOUT` | `30` | Seconds before half-open retry |
| `CIRCUIT_BREAKER_TIMEOUT` | `10` | Request timeout (seconds) |

## Service Discovery & etcd

| Variable | Default | Description |
|---|---|---|
| `ETCD_HOST` | `localhost` | etcd host |
| `ETCD_PORT` | `2379` | etcd port |
| `ETCD_PREFIX` | `/universal-translation/decoders/` | etcd key prefix |
| `SERVICE_TTL` | `60` | Service registration TTL (seconds) |
| `COORDINATOR_MIRROR_INTERVAL` | `30` | Redis→disk mirror interval |

## Prefetch (Decoder Startup)

| Variable | Default | Description |
|---|---|---|
| `PREFETCH_VOCAB_GROUPS` | — | Comma-separated vocab packs to preload |
| `PREFETCH_ADAPTERS` | — | Comma-separated adapters to preload |
| `PREFETCH_MODELS` | — | Comma-separated model artifacts to preload |

## Security / CORS / Proxies

| Variable | Default | Description |
|---|---|---|
| `ALLOWED_ORIGINS` | — | CORS allowed origins (comma-separated) |
| `TRUSTED_PROXIES` | — | Trusted proxy IPs |

## Training & Distributed

| Variable | Default | Description |
|---|---|---|
| `MASTER_ADDR` | `localhost` | Distributed training master address |
| `MASTER_PORT` | `12355` | Distributed training master port |
| `TRAIN_TOTAL_STEPS` | — | Total training steps (auto-compile hint) |
| `EVOLVE_ANALYTICS_JSON` | — | Path to vocab evolution analytics output |
| `TORCH_COMPILE_DEBUG` | `0` | Torch compile debug level |
| `TORCH_LOGS` | `+dynamo` | Torch log modules |
| `TORCHDYNAMO_DISABLE_CACHE_LIMIT` | `1` | Disable Dynamo cache limit |
| `TOKENIZERS_PARALLELISM` | `false` | Tokenizer parallelism toggle |
| `PYTORCH_CUDA_ALLOC_CONF` | `max_split_size_mb:512` | PyTorch CUDA allocator config |
| `OMP_NUM_THREADS` | `4` | OpenMP threads |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device selection |

## Universal Decoder Node (`udn`)

| Variable | Default | Description |
|---|---|---|
| `DECODER_ENDPOINT` | `http://localhost:8001` | Decoder endpoint for UDN |
| `DECODER_GPU_ENABLED` | auto-detected | Force GPU enable/disable |
| `DECODER_GPU` | auto-detected | GPU type: `cuda`, `mps`, `cpu` |
| `LOCAL_DECODER_URL` | `http://localhost:8001` | Local decoder URL for SDKs |
| `PREFER_LOCAL_DECODER` | `false` | SDK prefers local decoder over cloud |

## CICD & Publishing

| Variable | Description |
|---|---|
| `MAVEN_USERNAME` | Maven repository username |
| `MAVEN_PASSWORD` | Maven repository password |
| `MAVEN_URL` | Maven repository URL |
| `NPM_TOKEN` | npm publish token |
| `COCOAPODS_TRUNK_TOKEN` | CocoaPods trunk token |
| `PYPI_API_TOKEN` | PyPI API token |
| `DOCKERHUB_USERNAME` | Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub token |
| `DOCKERHUB_REPO` | Docker Hub repository |

## UTS-Prefixed Overrides (constants.py)

These accept both `UTS_<NAME>` and bare `<NAME>`. Unset variables fall through to defaults in `utils/constants.py`.

### Timing & Retries
| Variable | Default |
|---|---|
| `UTS_DEFAULT_TIMEOUT` | `30` |
| `UTS_MAX_RETRY_COUNT` | `3` |
| `UTS_DEFAULT_BATCH_SIZE` | `64` |
| `UTS_DEFAULT_BUFFER_SIZE` | `8192` |

### Caching & Resource Limits
| Variable | Default |
|---|---|
| `UTS_MAX_CACHE_SIZE` | `10000` |
| `UTS_DEFAULT_CACHE_TTL` | `3600` |
| `UTS_MAX_MEMORY_USAGE` | `1073741824` |
| `UTS_MAX_FILE_SIZE` | `104857600` |

### Security
| Variable | Default |
|---|---|
| `UTS_TOKEN_EXPIRATION` | `1800` |
| `UTS_REFRESH_TOKEN_EXPIRATION` | `604800` |
| `UTS_PASSWORD_MIN_LENGTH` | `8` |
| `UTS_PASSWORD_MAX_LENGTH` | `128` |
| `UTS_MAX_LOGIN_ATTEMPTS` | `5` |
| `UTS_LOCKOUT_DURATION` | `900` |

### Encoder Dimensions
| Variable | Default |
|---|---|
| `UTS_ENCODER_EMBEDDING_DIM` | `512` |
| `UTS_ENCODER_HIDDEN_DIM` | `1024` |
| `UTS_ENCODER_NUM_LAYERS` | `6` |
| `UTS_ENCODER_NUM_HEADS` | `8` |
| `UTS_ENCODER_DROPOUT` | `0.1` |
| `UTS_ENCODER_MAX_LENGTH` | `512` |

### Decoder Dimensions
| Variable | Default |
|---|---|
| `UTS_DECODER_EMBEDDING_DIM` | `512` |
| `UTS_DECODER_HIDDEN_DIM` | `1024` |
| `UTS_DECODER_NUM_LAYERS` | `6` |
| `UTS_DECODER_NUM_HEADS` | `8` |
| `UTS_DECODER_DROPOUT` | `0.1` |
| `UTS_DECODER_MAX_LENGTH` | `512` |

### Vocabulary
| Variable | Default |
|---|---|
| `UTS_VOCAB_SIZE` | `32000` |
| `UTS_VOCAB_MIN_FREQUENCY` | `5` |
| `UTS_VOCAB_SPECIAL_TOKENS` | `<pad>,<unk>,<bos>,<eos>` |
| `UTS_VOCAB_PAD_ID` | `0` |
| `UTS_VOCAB_UNK_ID` | `1` |
| `UTS_VOCAB_BOS_ID` | `2` |
| `UTS_VOCAB_EOS_ID` | `3` |

### API
| Variable | Default |
|---|---|
| `UTS_API_RATE_LIMIT` | `100` |
| `UTS_API_BURST_LIMIT` | `20` |
| `UTS_API_TIMEOUT` | `30` |
| `UTS_API_VERSION` | `1.0.0` |

### File Name Overrides
| Variable | Default |
|---|---|
| `UTS_ENCODER_MODEL_FILENAME` | `encoder.pt` |
| `UTS_DECODER_MODEL_FILENAME` | `decoder.pt` |
| `UTS_BEST_MODEL_FILENAME` | `best_model.pt` |
| `UTS_BASE_CONFIG_FILENAME` | `base.yaml` |
| `UTS_VERSION_CONFIG_FILENAME` | `version-config.json` |
| `UTS_BENCHMARK_RESULTS_FILENAME` | `benchmark_results.json` |
| `UTS_TRAINING_REPORT_FILENAME` | `training_report.json` |
| `UTS_EVALUATION_REPORT_FILENAME` | `evaluation_report.json` |
| `UTS_EMERGENCY_CHECKPOINT_FILENAME` | `emergency_checkpoint.pt` |
| `UTS_QUANTIZATION_REPORT_FILENAME` | `quantization_report.json` |
| `UTS_PROFILING_REPORT_FILENAME` | `profiling_report.json` |
| `UTS_SUPPORTED_VOCAB_FORMAT` | `1` |

## Secret Bootstrap & Validation

The system supports `*_FILE` environment variables for sensitive values, loaded at startup via `utils/secrets_bootstrap.py`. Resolution order:
1. `VAR_FILE` env var → read file content → set `VAR`
2. Direct `VAR` env var
3. Credential manager (`keyring` + encrypted file fallback)

At startup, `validate_runtime_secrets(role)` checks:
- All critical secrets have ≥32 chars
- No placeholder values are in use
- RS256 keys (if used) are ≥2048 bits
- Role-appropriate secrets are present

## Security Tips
1. Replace all defaults in production; store secrets in a vault
2. Prefer `*_FILE` pattern inside containers and orchestrators (Docker Secrets, K8s Secrets)
3. Terminate TLS at your ingress/proxy
4. Rotate JWT secrets every 90 days via `uts tools --rotate-secrets`
5. Set `UTS_TEST_MODE=1` in CI/test environments only

## Troubleshooting
- Verify envs inside containers: `docker compose exec <svc> env | sort`
- In K8s, check `kubectl describe deploy <name>` and ConfigMaps/Secrets mounting
- See `docs/TROUBLESHOOT.md` for more
