#!/bin/bash
# cloud_decoder/build_and_run.sh

# Build and run the decoder service

set -e

echo "🐋 Building Universal Decoder Docker image..."

# Build the image
docker build -t universal-decoder:latest .

# Create necessary directories
mkdir -p models vocabs logs

echo "📦 Checking for model files..."

# Check if model files exist
if [ ! -f "models/decoder_model.pt" ]; then
    echo "⚠️  Warning: No decoder model found at models/decoder_model.pt"
    echo "Please place your model file in the models/ directory"
fi

if [ ! -d "vocabs" ] || [ -z "$(ls -A vocabs)" ]; then
    echo "⚠️  Warning: No vocabulary packs found in vocabs/"
    echo "Please place your vocabulary packs in the vocabs/ directory"
fi

echo "🚀 Starting Universal Decoder service..."

# Run with docker-compose
docker-compose up -d

echo "✅ Service started!"
echo "📡 API available at: http://localhost:8000"
echo "📊 Health check: http://localhost:8000/health"
echo "📚 API docs: http://localhost:8000/docs"

# Show logs
echo -e "\n📜 Showing logs (Ctrl+C to exit)..."
docker-compose logs -f decoder