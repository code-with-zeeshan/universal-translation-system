[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repoRoot = "c:\Users\DELL\universal-translation-system"

function Assert-LastExit {
  param([string]$Message)
  if ($LASTEXITCODE -ne 0) { throw $Message }
}

Write-Host "🚀 Deploying Universal Translation System..."

# 1. Build Docker images
Write-Host "🐳 Building Docker images..."

docker build -f "$repoRoot\cloud_decoder\Dockerfile" -t universal-decoder:latest "$repoRoot"
Assert-LastExit "Decoder image build failed"

docker build -f "$repoRoot\docker\encoder.Dockerfile" -t universal-encoder-core:latest "$repoRoot"
Assert-LastExit "Encoder image build failed"

# 2. Push images to registry (optional)
# docker tag universal-decoder:latest <your-registry>/universal-decoder:latest
# docker push <your-registry>/universal-decoder:latest
# docker tag universal-encoder-core:latest <your-registry>/universal-encoder-core:latest
# docker push <your-registry>/universal-encoder-core:latest

# 3. Deploy to Kubernetes
Write-Host "☸️  Deploying to Kubernetes..."
kubectl apply -f "$repoRoot\kubernetes\namespace.yaml"; if ($LASTEXITCODE -ne 0) { throw "kubectl apply namespace failed" }
kubectl apply -f "$repoRoot\kubernetes\encoder-artifacts-pvc.yaml"; if ($LASTEXITCODE -ne 0) { throw "kubectl apply pvc failed" }
kubectl apply -f "$repoRoot\kubernetes\encoder-build.yaml"; if ($LASTEXITCODE -ne 0) { throw "kubectl apply encoder-build failed" }
kubectl apply -f "$repoRoot\kubernetes\decoder-deployment.yaml"; if ($LASTEXITCODE -ne 0) { throw "kubectl apply decoder deployment failed" }
kubectl apply -f "$repoRoot\kubernetes\decoder-service.yaml"; if ($LASTEXITCODE -ne 0) { throw "kubectl apply decoder service failed" }

Write-Host "✅ Deployment complete!"