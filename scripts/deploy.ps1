[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot

function Assert-LastExit {
  param([string]$Message)
  if ($LASTEXITCODE -ne 0) { throw $Message }
}

Write-Host "Deploying Universal Translation System..."

# 1. Build Docker images
Write-Host "Building Docker images..."

docker build -f "$repoRoot\docker\decoder.Dockerfile" -t universal-decoder:latest "$repoRoot"
Assert-LastExit "Decoder image build failed"

docker build -f "$repoRoot\docker\encoder.Dockerfile" -t universal-encoder-core:latest "$repoRoot"
Assert-LastExit "Encoder image build failed"

docker build -f "$repoRoot\docker\coordinator.Dockerfile" -t universal-coordinator:latest "$repoRoot"
Assert-LastExit "Coordinator image build failed"

# 2. Push images to registry (optional)
# docker tag universal-decoder:latest <your-registry>/universal-decoder:latest
# docker push <your-registry>/universal-decoder:latest
# docker tag universal-encoder-core:latest <your-registry>/universal-encoder-core:latest
# docker push <your-registry>/universal-encoder-core:latest

# 3. Deploy to Kubernetes (Helm preferred)
Write-Host "Deploying via Helm..."
helm upgrade --install uts "$repoRoot\charts\uts" --namespace uts-prod --create-namespace

Write-Host "Deployment complete!"
