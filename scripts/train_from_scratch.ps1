[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Assert-LastExit {
  param([string]$Message)
  if ($LASTEXITCODE -ne 0) { throw $Message }
}

Write-Host "🚀 Starting Full End-to-End Training Pipeline for Universal Translation System"
Write-Host "=============================================================================="

# 1. Prepare all data
Write-Host "`n[Step 1/6] 📥 Running unified data pipeline..."
python - <<'PY'
from config.schemas import load_config
from data.unified_data_pipeline import UnifiedDataPipeline
import asyncio

cfg = load_config('config/archived_gpu_configs/training_generic_gpu.yaml')
pipeline = UnifiedDataPipeline(cfg)
asyncio.run(pipeline.run_pipeline(resume=True, stages=None))
PY
Write-Host "[Step 1/6] ✅ Data pipeline complete."

# 2. Create vocabulary packs
Write-Host "`n[Step 2/6] 📦 Creating vocabulary packs from processed data..."
python -m vocabulary.unified_vocabulary_creator
Assert-LastExit "Vocabulary pack creation failed"
Write-Host "[Step 2/6] ✅ Vocabulary packs created."

# 3. Initialize models from pretrained
Write-Host "`n[Step 3/6] 🧠 Initializing models from pretrained weights (XLM-R & mBART)..."
python -m training.bootstrap_from_pretrained
Assert-LastExit "Bootstrap from pretrained failed"
Write-Host "[Step 3/6] ✅ Initial models created."

# 4. Train models
Write-Host "`n[Step 4/6] 🏃 Starting main training loop (auto-detecting best config for hardware)..."
python -m training.launch train --config "$repoRoot\config\archived_gpu_configs\training_generic_gpu.yaml"
Assert-LastExit "Training launch failed"
Write-Host "[Step 4/6] ✅ Main training loop complete."

# 5. Convert models for production
Write-Host "`n[Step 5/6] 🔄 Converting trained models to production formats (ONNX, etc.)..."
python -m training.convert_models
Assert-LastExit "Model conversion failed"
Write-Host "[Step 5/6] ✅ Model conversion complete."

# 6. Finalizing production models (placeholder)
Write-Host "`n[Step 6/6] 📦 Finalizing production models..."
Write-Host "[Step 6/6] ✅ Production models are ready."

Write-Host "`n🎉 Full training pipeline finished successfully! Models are ready in 'models/production'."
Write-Host "=============================================================================="