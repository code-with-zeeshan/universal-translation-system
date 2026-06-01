# UTS — Fresh Studio Setup Commands

## 0. Open a terminal in Lightning AI Studio

All commands run in the **terminal** (not notebook). The environment already has:
- Python 3.12, CUDA 12.8, PyTorch 2.8.0
- conda (default env is `base`)

---

## Step 1 — Clone

```bash
cd /teamspace/studios/this_studio
git clone https://github.com/code-with-zeeshan/universal-translation-system.git
cd universal-translation-system
```

**Troubleshoot:**
- `git: command not found` → Studio is too bare. Run `conda install git -y`
- `Permission denied` → You're outside `/teamspace`. Use `cd /teamspace/studios/this_studio` first

---

## Step 2 — Install deps

```bash
pip install -e . 2>&1 | tail -20
```

**Troubleshoot:**
- `pip: command not found` → Run `conda install pip -y`
- `error: externally-managed-environment` → Run `conda install pip -y` to get conda's pip
- `ERROR: Could not find a version that satisfies ...` → Single dep issue. Run `pip install <pkg>` separately
- Takes ~5 min. Ignore dependency conflicts (they're non-fatal)

---

## Step 3 — Install LitServe (for decoder, not needed for pipeline)

```bash
pip install "litserve>=0.12.0" 2>&1 | tail -10
```

**Troubleshoot:**
- Only needed if you run the decoder via LitServe. Pipeline doesn't need it.

---

## Step 4 — Export HMAC key (required!)

```bash
export UTS_HMAC_KEY="dev-only-change-in-production-1234567890abc"
```

**Must be set in the same terminal session before every `python -m` command.** Add to `~/.bashrc` to auto-set:

```bash
echo 'export UTS_HMAC_KEY="dev-only-change-in-production-1234567890abc"' >> ~/.bashrc
source ~/.bashrc
```

**Troubleshoot:**
- `ValueError: HMAC key not configured` → You forgot to export the variable
- `KeyError: 'UTS_HMAC_KEY'` → Same thing

---

## Step 5 — Run core pipeline (~25 min)

```bash
python -m data.unified_data_pipeline --config config/base.yaml
```

**Expected output last line:**
```
Pipeline completed in 0.XX hours
Total data: XXX,XXX sentences (X.XX GB)
```

**Troubleshoot:**

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'yaml'` | Missing PyYAML | `pip install pyyaml` |
| `ModuleNotFoundError: No module named 'torch'` | Wrong Python env | Studio should have torch. Run `conda install pytorch -c pytorch -y` |
| `ModuleNotFoundError: No module named 'transformers'` | Missing HF lib | `pip install transformers` |
| `ModuleNotFoundError: No module named 'datasets'` | Missing HF datasets | `pip install datasets` |
| `HTTPError: 403` or `HTTPError: 401` | Cannot reach Hugging Face hub | Check internet: `curl -I https://huggingface.co`. If blocked, use `--config config/offline.yaml` |
| `ConnectionError` / timeout | Network issue or HF hub down | Retry. `--resume` picks up where it left off |
| `CUDA out of memory` | NLLB-1.3B too big for GPU | Edit `config/base.yaml`: change `base_model` to `facebook/nllb-200-distilled-600M` |
| `*_sampled.txt not found` in later stages | A prior stage failed | Check logs in `data/log/`. Re-run with `--resume` |
| Pipeline seems stuck at "Downloading..." | HF dataset download in progress | Let it finish. Check `data/raw/*.txt` growing |
| `ValueError: I/O operation on closed file` | Rare race in logging | Re-run. If persistent, `pip install --upgrade tenacity` |

**To resume after a failure:**
```bash
python -m data.unified_data_pipeline --config config/base.yaml --resume
```

**To reset pipeline state (start fresh):**
```bash
python -m data.unified_data_pipeline --config config/base.yaml --reset
rm -rf data/raw data/processed
```

**To run a single stage:**
```bash
python -m data.unified_data_pipeline --config config/base.yaml --stage sample_filter
```

---

## Step 6 — Verify output

```bash
# Count total training sentences
wc -l data/processed/train_final.txt

# Check language coverage
ls data/processed/ready/*_corpus.txt

# Check vocabulary files
ls -la vocabulary/*.model
```

**Expected:**
- `train_final.txt`: ~4-5 million lines
- `*_corpus.txt`: one per language (20 files)
- `vocabulary/*.model`: 6 SentencePiece models (latin, cjk, arabic, devanagari, cyrillic, thai)

---

## Step 7 — Train (~2-4 hours on T4)

```bash
python -m training.intelligent_trainer --config config/base.yaml
```

**Troubleshoot:**
- `CUDA out of memory` → In `config/base.yaml`, reduce `batch_size: 16`, enable `cpu_offload: true`
- `ValueError: Expected input batch_size (...) to match target batch_size (...)` → Mixed sequence lengths. Re-run pipeline with `max_sentence_length: 64` in config
- Training loss is NaN → Reduce `lr: 1e-4` in config

---

## Step 8 — Evaluate

```bash
python main.py --mode evaluate
```

---

## Quick reference

```bash
# One-shot: clone + install + export + run
cd /teamspace/studios/this_studio && \
git clone https://github.com/code-with-zeeshan/universal-translation-system.git && \
cd universal-translation-system && \
pip install -e . && \
export UTS_HMAC_KEY="dev-only-change-in-production-1234567890abc" && \
python -m data.unified_data_pipeline --config config/base.yaml

# After that, train:
python -m training.intelligent_trainer --config config/base.yaml
```

---

## Where to find logs

| What | Where |
|---|---|
| Pipeline logs | `data/log/data.log` |
| Training logs | `logs/training/` |
| Config used | `config/base.yaml` |
| Pipeline state | `data/pipeline_checkpoint.json` |
| Model checkpoints | `checkpoints/default_run/` |
