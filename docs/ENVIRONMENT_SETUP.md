# Environment Setup

## Colab (T4 GPU)

```python
!git clone https://github.com/code-with-zeeshan/universal-translation-system.git
%cd universal-translation-system
!pip install -e ".[data]"
!pip install flash-attn optimum --quiet   # optional: 2-3x NLLB speedup
!hf auth login
!python -m scripts.init_env --role general
!uts data --pipeline --config config/base.yaml
```

**Notes:**
- Colab pre-installs CUDA + PyTorch — no extra driver setup needed
- `pip install -e ".[data]"` may print dependency conflict warnings (numpy, protobuf, fsspec, etc.). These are **warnings, not errors** — the pipeline runs fine. Colab ships many ML packages whose exact versions conflict with `unbabel-comet`'s transitive deps (pytorch-lightning, protobuf). Ignore them.
- Restart runtime after `pip install` if it warns about previously imported packages (`WARNING: The following packages were previously imported in this runtime`).
- T4 GPU auto-detected → optimal batch sizes/workers applied automatically.

## Lightning AI (L4 / L40S / A100)

```bash
git clone https://github.com/code-with-zeeshan/universal-translation-system.git
cd universal-translation-system
pip install -e ".[data]"
pip install flash-attn optimum   # recommended: NLLB speedup
huggingface-cli login
python -m scripts.init_env --role general
uts data --pipeline --config config/base.yaml
```

**Notes:**
- Lightning AI already has CUDA + PyTorch — no extra setup
- L4 is auto-detected as L4, L40S as L40S, A100 as A100 — each with appropriate batch sizes
- `torch.compile` enabled on L4+ (not on T4)

## Local / GPU-Enabled IDE

```bash
# Create virtual environment first
python -m venv .venv && source .venv/bin/activate

pip install -e ".[data]"
pip install flash-attn optimum   # optional accelerators
huggingface-cli login
python -m scripts.init_env --role general
uts data --pipeline --config config/base.yaml
```

**Notes:**
- Virtual environment avoids all system-level dependency conflicts
- `flash-attn` requires CUDA + compatible GPU (compute 7.5+)
- `UTS_GPU_TIER` env var available to force a specific profile

## If You See Dependency Warnings

Warnings like these are **safe to ignore** — the packages still work:

```
google-colab 1.0.0 requires requests==2.32.4, but you have requests 2.34.2
tensorflow 2.20.0 requires protobuf>=5.28.0, but you have protobuf 4.25.9
numpy 2.0.2 is installed but numpy<2.0.0 is required by universal-translation-system
```

The pipeline's dependencies (`torch`, `transformers`, `sentence-transformers`, `unbabel-comet`) are all satisfied — the warnings only affect unrelated Colab pre-installed packages (tensorflow, jax, opencv, etc.).

If you want a clean environment with zero warnings, use a virtual environment or Docker.
