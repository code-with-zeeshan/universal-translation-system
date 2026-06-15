#!/bin/bash
set -e

# scripts/setup_serving.sh - Configure serving infrastructure
# Usage: bash scripts/setup_serving.sh [--cloud] [--node] [--all]
#
# Two serving modes:
#   1. cloud_decoder — FastAPI service (runs on cloud, serves encoder traffic)
#   2. universal-decoder-node — Local node (runs encoder+decoder locally, registers to pool)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

check_python_deps() {
    echo "--- Checking Python dependencies ---"
    local missing=0

    for pkg in fastapi uvicorn litserve torch; do
        if python3 -c "import $pkg" 2>/dev/null; then
            echo "  $pkg: OK"
        else
            echo "  $pkg: MISSING"
            missing=1
        fi
    done

    if [ "$missing" -eq 1 ]; then
        echo ""
        echo "Install missing dependencies:"
        echo "  pip install -r $PROJECT_DIR/requirements/base.txt"
        echo "  pip install -r $PROJECT_DIR/requirements/serve.txt"
        echo "  pip install -r $PROJECT_DIR/requirements/decoder.txt"
        echo ""
        echo "Or install everything at once:"
        echo "  pip install -r $PROJECT_DIR/requirements/base.txt \\"
        echo "      -r $PROJECT_DIR/requirements/serve.txt \\"
        echo "      -r $PROJECT_DIR/requirements/decoder.txt"
        return 1
    fi

    return 0
}

setup_cloud() {
    echo "=== Setting up Cloud Decoder (FastAPI) ==="
    echo ""

    check_python_deps || true

    local app_path="runtime.cloud_decoder.optimized_decoder:app"

    echo ""
    echo "=== Cloud Decoder — Starting Instructions ==="
    echo ""
    echo "  # Start with uvicorn (single worker):"
    echo "  cd $PROJECT_DIR"
    echo "  uvicorn $app_path --host 0.0.0.0 --port 8000"
    echo ""
    echo "  # Start with uvicorn (multi-worker):"
    echo "  uvicorn $app_path --host 0.0.0.0 --port 8000 --workers 4"
    echo ""
    echo "  # Start with litserve (recommended for production):"
    echo "  litserve serve $app_path --host 0.0.0.0 --port 8000"
    echo ""
    echo "  # Or run directly:"
    echo "  python runtime/cloud_decoder/optimized_decoder.py"
    echo ""
    echo "  # Docker:"
    echo "  docker build -f docker/Dockerfile.cloud -t uts-cloud-decoder ."
    echo "  docker run --gpus all -p 8000:8000 uts-cloud-decoder"
    echo ""

    local env_example="$PROJECT_DIR/.env.example"
    if [ -f "$env_example" ]; then
        echo "See $env_example for environment configuration."
    fi
}

setup_node() {
    echo "=== Setting up Universal Decoder Node (Python CLI) ==="
    echo ""

    if ! command -v python3 &>/dev/null; then
        echo "ERROR: Python 3 is required. Install Python 3.8+ first."
        exit 1
    fi

    local node_dir="$PROJECT_DIR/universal-decoder-node"
    if [ ! -d "$node_dir" ]; then
        echo "ERROR: universal-decoder-node directory not found at $node_dir"
        exit 1
    fi

    echo "--- Installing universal-decoder-node package ---"
    echo ""

    pip install -e "$node_dir"

    echo ""
    echo "=== Verify installation ==="
    echo ""
    echo "  udn --help"
    echo ""

    echo "=== Starting the node ==="
    echo ""
    echo "  udn start --port 8000"
    echo ""
    echo "  # Or with a config file:"
    echo "  udn init --output config.yaml"
    echo "  udn start --config config.yaml"
    echo ""

    echo "=== Registering with coordinator ==="
    echo ""
    echo "  # Interactive registration:"
    echo "  udn register"
    echo ""
    echo "  # Non-interactive (save to file):"
    echo "  udn register \\"
    echo "      --name my-node \\"
    echo "      --endpoint https://my-node.example.com \\"
    echo "      --region us-east-1 \\"
    echo "      --gpu-type T4 \\"
    echo "      --capacity 100 \\"
    echo "      --output registration.json"
    echo ""
    echo "  # Register with coordinator directly:"
    echo "  udn register \\"
    echo "      --name my-node \\"
    echo "      --endpoint https://my-node.example.com \\"
    echo "      --coordinator-url http://coordinator:5100"
    echo ""

    echo "=== Testing the node ==="
    echo ""
    echo "  universal-decoder-node test \\"
    echo "      --text 'Hello world' \\"
    echo "      --source-lang en \\"
    echo "      --target-lang es \\"
    echo "      --endpoint http://localhost:8000"
    echo ""
}

show_env_guide() {
    cat <<EOF
=== Environment Variable Configuration Guide ===

Required:
  None — all variables have defaults or fallbacks.

Cloud Decoder (optimized_decoder.py):
  API_HOST              Host to bind (default: 0.0.0.0)
  API_PORT              Port to bind (default: 8000)
  API_WORKERS           Number of uvicorn workers (default: 1)
  MODEL_VERSION         Model version string (default: 1.0.0)
  DECODER_CONFIG_PATH   Path to decoder config YAML (default: config/decoder_config.yaml)
  HF_HUB_REPO_ID        Hugging Face Hub repo for model artifacts
  DECODER_JWT_SECRET    JWT secret for auth (or set via secrets bootstrap)
  ALLOWED_ORIGINS       CORS allowed origins (comma-separated)
  TRUSTED_PROXIES       Trusted proxy IPs (comma-separated)
  LOG_LEVEL             Logging level (default: INFO)
  REDIS_URL             Redis connection URL (optional, fallback to disk)
  OMP_NUM_THREADS       OpenMP threads for PyTorch (default: 4)

Universal Decoder Node:
  DECODER_HOST          Host to bind (default: 0.0.0.0)
  DECODER_PORT          Port to bind (default: 8000)
  DECODER_WORKERS       Number of workers (default: 1)
  DECODER_ENDPOINT      Public endpoint URL for registration
  COORDINATOR_URL       Coordinator service URL (default: http://localhost:5100)
  COORDINATOR_TOKEN     Auth token for coordinator registration
  VOCAB_DIR             Vocabulary directory (default: vocabs)
  MODEL_PATH            Path to model file
  JWT_SECRET            JWT secret for auth

Redis:
  UTS_REDIS_URL         Redis connection URL (e.g., redis://localhost:6379/0)
  REDIS_KEY_PREFIX      Key prefix (default: translation:)
  REDIS_CONN_TIMEOUT    Connection timeout in seconds (default: 2)
  REDIS_READ_TIMEOUT    Read timeout in seconds (default: 2)

Secrets (via utils/secrets_bootstrap.py):
  DECODER_JWT_SECRET    JWT secret for decoder
  COORDINATOR_TOKEN     Token for coordinator auth
  REDIS_URL             Redis connection URL
EOF
}

case "${1:-}" in
    --cloud)  setup_cloud ;;
    --node)   setup_node ;;
    --all)
        setup_cloud
        echo ""
        echo "============================================"
        echo ""
        setup_node
        echo ""
        echo "============================================"
        echo ""
        show_env_guide
        ;;
    --env)
        show_env_guide
        ;;
    *)
        cat <<EOF
Usage: bash scripts/setup_serving.sh [--cloud] [--node] [--all] [--env]

Options:
  --cloud    Check and configure cloud decoder (FastAPI)
  --node     Check and configure universal decoder node (Python CLI)
  --all      Configure both serving modes
  --env      Print environment variable configuration guide
EOF
        exit 1
        ;;
esac
