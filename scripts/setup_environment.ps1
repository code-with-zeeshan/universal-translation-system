[CmdletBinding()]
param(
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = 'Stop'
$repoRoot = "c:\Users\DELL\universal-translation-system"
Set-Location $repoRoot

Write-Host "🔧 Setting up Universal Translation System environment..."

# 1. Create and activate Python virtual environment
Write-Host "Creating virtual environment at $VenvPath"
python -m venv $VenvPath

# Activate venv for current session
$venvActivate = Join-Path $VenvPath "Scripts\Activate.ps1"
. $venvActivate

# 2. Upgrade pip and install core requirements
python -m pip install --upgrade pip
python -m pip install -r "$repoRoot\requirements.txt"

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