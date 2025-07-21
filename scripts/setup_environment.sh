#!/bin/bash
# scripts/setup_environment.sh

set -e

echo "🔧 Setting up Universal Translation System environment..."

# 1. Create and activate Python virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Upgrade pip and install core requirements
pip install --upgrade pip
pip install -r requirements.txt

# 3. Install additional system dependencies (if needed)
# For Ubuntu/Debian:
# sudo apt-get update && sudo apt-get install -y build-essential cmake git wget lz4 liblz4-dev

# 4. Check CUDA installation
if command -v nvidia-smi &> /dev/null; then
    echo "✅ CUDA detected: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
else
    echo "⚠️  CUDA not detected. GPU training will not be available."
fi

echo "✅ Environment setup complete!"
