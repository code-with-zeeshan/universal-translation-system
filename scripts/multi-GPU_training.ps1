[CmdletBinding()]
param(
    [int]$NumGpus
)

$ErrorActionPreference = 'Stop'

# Auto-detect number of GPUs if not provided
if (-not $PSBoundParameters.ContainsKey('NumGpus')) {
    try {
        $gpuList = & nvidia-smi -L 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $gpuList) {
            throw "nvidia-smi not available"
        }
        $NumGpus = ($gpuList | Measure-Object -Line).Lines
    } catch {
        Write-Error "Could not detect GPUs. Install NVIDIA drivers and CUDA Toolkit, or pass -NumGpus explicitly."
        exit 1
    }
}

# Set CUDA_VISIBLE_DEVICES
$devices = (0..($NumGpus-1)) -join ','
$env:CUDA_VISIBLE_DEVICES = $devices

Write-Host "🚀 Starting distributed training on $NumGpus GPUs..."

# Prefer torchrun (PyTorch>=1.10)
$torchrun = Get-Command torchrun -ErrorAction SilentlyContinue
if (-not $torchrun) {
    Write-Error "torchrun not found. Ensure PyTorch is installed (pip install torch) and torchrun is on PATH."
    exit 1
}

# Use absolute path to repo to avoid cd chaining
$repoRoot = "c:\Users\DELL\universal-translation-system"

# Run distributed training
& torchrun --nproc_per_node $NumGpus --master_port 29500 -m training.launch train --distributed --config "$repoRoot\config\archived_gpu_configs\training_generic_multi_gpu.yaml"

# Optionally monitor GPU usage
try {
    Write-Host "🔍 Monitoring GPU usage... (Press Ctrl+C to exit)"
    & powershell -NoExit -Command "while ($true) { nvidia-smi; Start-Sleep -Seconds 1 }"
} catch {
    Write-Warning "nvidia-smi watch loop ended."
}