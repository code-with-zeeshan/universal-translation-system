# Environment Variables

This document provides a comprehensive list of all environment variables used in the Universal Translation System. These variables can be configured to customize the behavior of various components.

## How to Use Environment Variables

There are several ways to set environment variables:

1. **Using a .env file**:
   - Copy the `.env.example` file to `.env` in the project root
   - Modify the values as needed
   - The system will automatically load these values at runtime

2. **Setting in your shell**:
   ```bash
   # Linux/macOS
   export DECODER_API_URL=https://your-custom-domain.com/decode
   
   # Windows PowerShell
   $env:DECODER_API_URL = "https://your-custom-domain.com/decode"
   ```

3. **In Docker Compose**:
   - Environment variables can be set in the `docker-compose.yml` file
   - You can also create a `.env` file in the same directory as your `docker-compose.yml`

4. **In Kubernetes**:
   - Use ConfigMaps and Secrets to manage environment variables
   - See `kubernetes/` directory for example configurations

## General Configuration

| Variable | Description | Default Value |
|----------|-------------|---------------|
| `MODEL_VERSION` | Version of the model being used | `1.0.0` |

## Web SDK Configuration

| Variable | Description | Default Value |
|----------|-------------|---------------|
| `DECODER_API_URL` | URL for the decoder API | `https://api.yourdomain.com/decode` |
| `ENCODER_API_URL` | URL for the encoder API | `https://api.universal-translation.com/encode` |
| `MODEL_URL` | Path to the model file | `/models/universal_encoder.onnx` |
| `VOCAB_URL` | Path to vocabulary files | `/vocabs` |
| `WASM_ENCODER_PATH` | Path to WebAssembly encoder | `/wasm/encoder.js` |
| `USE_WASM_ENCODER` | Whether to use WebAssembly encoder | `true` |
| `ENABLE_FALLBACK` | Enable fallback to cloud API | `true` |

## React Native SDK Configuration

| Variable | Description | Default Value |
|----------|-------------|---------------|
| `USE_NATIVE_ENCODER` | Whether to use native encoder | `true` |

## Cloud Decoder Configuration

| Variable | Description | Default Value |
|----------|-------------|---------------|
| `API_HOST` | Host to bind the API server | `0.0.0.0` |
| `API_PORT` | Port for the API server | `8000` |
| `API_WORKERS` | Number of worker processes | `1` |
| `API_TITLE` | Title for the API documentation | `Cloud Decoder API` |
| `DECODER_JWT_SECRET` | Secret key for JWT authentication | `jwtsecret123` |
| `DECODER_CONFIG_PATH` | Path to decoder configuration | `config/decoder_config.yaml` |
| `HF_HUB_REPO_ID` | Hugging Face Hub repository ID | `your-hf-org/universal-translation-system` |

## Coordinator Configuration

| Variable | Description | Default Value |
|----------|-------------|---------------|
| `COORDINATOR_HOST` | Host to bind the coordinator | `0.0.0.0` |
| `COORDINATOR_PORT` | Port for the coordinator | `5100` |
| `COORDINATOR_WORKERS` | Number of worker processes | `1` |
| `COORDINATOR_TITLE` | Title for the coordinator API | `Universal Translation Coordinator` |
| `COORDINATOR_SECRET` | Secret key for cookies | `a-very-secret-key-for-cookies` |
| `COORDINATOR_JWT_SECRET` | Secret key for JWT authentication | `a-super-secret-jwt-key` |
| `COORDINATOR_TOKEN` | Admin token for coordinator | `changeme123` |
| `INTERNAL_SERVICE_TOKEN` | Token for internal service auth | `internal-secret-token-for-service-auth` |
| `POOL_CONFIG_PATH` | Path to decoder pool configuration | `configs/decoder_pool.json` |

## Universal Decoder Node Configuration

| Variable | Description | Default Value |
|----------|-------------|---------------|
| `DECODER_ENDPOINT` | Endpoint for the decoder | `http://localhost:8000` |
| `COORDINATOR_URL` | URL for the coordinator | `http://localhost:5100` |
| `DECODER_HOST` | Host to bind the decoder | `0.0.0.0` |
| `DECODER_PORT` | Port for the decoder | `8000` |
| `DECODER_WORKERS` | Number of worker processes | `1` |
| `VOCAB_DIR` | Directory for vocabulary files | `vocabs` |

## Vocabulary Creation Configuration

| Variable | Description | Default Value |
|----------|-------------|---------------|
| `ENCODER_MODEL_PATH` | Path to encoder model | `models/production/encoder.pt` |
| `FALLBACK_MODEL_PATH` | Path to fallback model | `models/fallback/encoder.pt` |
| `EMBEDDING_DIM` | Dimension of embeddings | `768` |

## Docker Configuration

| Variable | Description | Default Value |
|----------|-------------|---------------|
| `ENCODER_PORT` | Port for the encoder service | `8000` |
| `DECODER_PORT` | Port for the decoder service | `8001` |
| `COORDINATOR_PORT` | Port for the coordinator service | `8002` |
| `PROMETHEUS_PORT` | Port for Prometheus | `9090` |
| `GRAFANA_PORT` | Port for Grafana | `3000` |
| `GRAFANA_ADMIN_PASSWORD` | Admin password for Grafana | `admin` |
| `DECODER_POOL` | Decoder pool configuration | `decoder:8001` |

## Security Recommendations

For production environments, we strongly recommend:

1. **Change all default secrets and tokens**
2. **Use environment-specific values for different deployments**
3. **Store sensitive values in secure vaults or secret managers**
4. **Rotate JWT secrets periodically**
5. **Use HTTPS for all API endpoints**

## Troubleshooting

If you encounter issues with environment variables:

1. Verify that the variables are correctly set using:
   ```bash
   # Linux/macOS
   printenv | grep DECODER
   
   # Windows PowerShell
   Get-ChildItem Env: | Where-Object { $_.Name -like "*DECODER*" }
   ```

2. Check that the application is loading from the expected location
3. Ensure Docker containers have the correct environment variables passed to them
4. Look for typos in variable names (they are case-sensitive)

For more help, see the [Troubleshooting Guide](troubleshooting.md).