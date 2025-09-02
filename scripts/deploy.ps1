[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

Write-Host "🚀 Deploying Universal Translation System..."

# 1. Build Docker images
Write-Host "🐳 Building Docker images..."

docker build -f "c:\Users\DELL\universal-translation-system\cloud_decoder\Dockerfile" -t universal-decoder:latest "c:\Users\DELL\universal-translation-system"

docker build -f "c:\Users\DELL\universal-translation-system\docker\encoder.Dockerfile" -t universal-encoder-core:latest "c:\Users\DELL\universal-translation-system"

# 2. Push images to registry (optional)
# docker tag universal-decoder:latest <your-registry>/universal-decoder:latest
# docker push <your-registry>/universal-decoder:latest
# docker tag universal-encoder-core:latest <your-registry>/universal-encoder-core:latest
# docker push <your-registry>/universal-encoder-core:latest

# 3. Deploy to Kubernetes
Write-Host "☸️  Deploying to Kubernetes..."
kubectl apply -f "c:\Users\DELL\universal-translation-system\kubernetes\namespace.yaml"
kubectl apply -f "c:\Users\DELL\universal-translation-system\kubernetes\encoder-artifacts-pvc.yaml"
kubectl apply -f "c:\Users\DELL\universal-translation-system\kubernetes\encoder-build.yaml"
kubectl apply -f "c:\Users\DELL\universal-translation-system\kubernetes\decoder-deployment.yaml"
kubectl apply -f "c:\Users\DELL\universal-translation-system\kubernetes\decoder-service.yaml"

Write-Host "✅ Deployment complete!"