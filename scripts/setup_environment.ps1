[CmdletBinding()]
param(
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Assert-LastExit {
  param([string]$Message)
  if ($LASTEXITCODE -ne 0) { throw $Message }
}

Write-Host "🔧 Setting up Universal Translation System environment..."

# 1. Create and activate Python virtual environment
Write-Host "Creating virtual environment at $VenvPath"
python -m venv $VenvPath
Assert-LastExit "Python venv creation failed"

# Activate venv for current session
$venvActivate = Join-Path $VenvPath "Scripts\Activate.ps1"
. $venvActivate

# 2. Upgrade pip and install core requirements
python -m pip install --upgrade pip
Assert-LastExit "pip upgrade failed"
# Base runtime
python -m pip install -r "$repoRoot\requirements\base.txt"
Assert-LastExit "base requirements install failed"
# Training + serving
python -m pip install -r "$repoRoot\requirements\train.txt" -r "$repoRoot\requirements\serve.txt"
Assert-LastExit "train/serve requirements install failed"
# Optional service-specific extras
# python -m pip install -r "$repoRoot\requirements\decoder.txt" -r "$repoRoot\requirements\coordinator.txt"

# 3. Optional: Additional system checks
try {
    $nv = & nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0 -and $nv) {
        Write-Host "✅ CUDA detected: $nv"
    } else {
        Write-Warning "⚠️  CUDA not detected. GPU training will not be available."
    }
} catch {
    Write-Warning "⚠️  CUDA not detected. GPU training will not be available."
}

Write-Host "✅ Environment setup complete!"}