#!/bin/bash
# scripts/deploy.sh

set -e

echo "🚀 Deploying Universal Translation System..."

# 1. Build Docker images
echo "🐳 Building Docker images..."
docker build -f cloud_decoder/Dockerfile -t universal-decoder:latest .
docker build -f docker/encoder.Dockerfile -t universal-encoder-core:latest .

# 2. Push images to registry (optional)
# docker tag universal-decoder:latest <your-registry>/universal-decoder:latest
# docker push <your-registry>/universal-decoder:latest
# docker tag universal-encoder-core:latest <your-registry>/universal-encoder-core:latest
# docker push <your-registry>/universal-encoder-core:latest

# 3. Deploy to Kubernetes
echo "☸️  Deploying to Kubernetes..."
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/encoder-artifacts-pvc.yaml
kubectl apply -f kubernetes/encoder-build.yaml
kubectl apply -f kubernetes/decoder-deployment.yaml
kubectl apply -f kubernetes/decoder-service.yaml

echo "✅ Deployment complete!"
