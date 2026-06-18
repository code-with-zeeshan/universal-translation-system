# Troubleshooting

## Setup & Installation

### `pip install -e .` fails
```
error: externally-managed-environment
```
**Fix:** `conda install pip -y` then retry.

### `git: command not found`
**Fix:** `conda install git -y`

### `No module named 'yaml'`
**Fix:** `pip install pyyaml` or `pip install -e ".[train]"`

### `HMAC key not configured`
```
ValueError: HMAC key not configured
```
**Fix:** `python scripts/init_env.py --role general` (auto-generates `.env` with strong `UTS_HMAC_KEY`)

---

## Data Pipeline

### `HTTPError: 403 / 401` during download
Cannot reach Hugging Face hub.
**Fix:** Check internet: `curl -I https://huggingface.co`. If blocked, use offline mode.

### `ConnectionError` / timeout during download
Network issue or HF hub down.
**Fix:** Retry with `--resume`.

### `*_sampled.txt not found` in later stages
A prior stage failed silently.
**Fix:** Check logs in `data/log/`. Re-run with `--resume`.

### CUDA out of memory during pipeline
NLLB model (used for backtranslation) is too big.
**Fix:** Edit config: change `base_model` to `facebook/nllb-200-distilled-600M`

### `ValueError: I/O operation on closed file`
Rare race condition in logging.
**Fix:** Re-run. If persistent, `pip install --upgrade tenacity`.

---

## Training

### Loss stays at ~11.0 (doesn't decrease)
The model isn't learning. Most common causes:
- **`use_lora: true`** — 95% of params are frozen and random. Set `use_lora: false`.
- **LR too low/high** — try `3e-4` for full training, `5e-4` for LoRA
- **Data issue** — check `train_final.txt` has content

### Loss = NaN
Learning rate too high or data contains bad values.
**Fix:** Reduce `lr` to `1e-4`, check data for NaN/inf.

### Validation loss > training loss
Normal sign of overfitting, but gap should be small.
**Fix:** Add more data, reduce epochs, or increase dropout.

### `CUDA out of memory`
```
RuntimeError: CUDA out of memory. Tried to allocate ...
```
**Fix:** 
- Reduce `batch_size` (e.g., 32 → 16)
- Enable `gradient_checkpointing: true`
- Reduce `accumulation_steps` (e.g., 4 → 2)

### `Expected input batch_size (...) to match target batch_size (...)`
Mixed sequence lengths in batch.
**Fix:** In config, set `max_sentence_length: 64` and re-run data pipeline.

### Training is very slow
**Fix:**
- Set `compile_model: true` (2-3× speedup)
- Set `mixed_precision: true` and `dtype: bfloat16`
- Set `flash_attention: true`
- Reduce `warmup_steps` (1000 is enough)

### `ValueError: The specified target language adapter 'xx' not found`
Missing target language adapter in decoder.
**Fix:** Ensure the language code is in `active_languages` in config, and rerun training to create adapters.

---

## Evaluation

### BLEU reported as 0.000 or N/A
The model produces near-random output. See "Loss stays at ~11.0" above.

### `KeyError: 'base_model.model.encoder.embedding_layer.weight'` during eval
Checkpoint key mismatch. The evaluator handles PEFT-wrapped models, but if the checkpoint was saved without LoRA, the keys differ.
**Fix:** Use `./uts eval --model --checkpoint path/to/checkpoint`

### `target_language_adapters.* keys not found in checkpoint`
Harmless warning. These adapters are created at runtime but weren't saved in older checkpoints.

---

## Serving

### `Address already in use`
Port is occupied.
**Fix:** Change port in config or kill existing process: `kill $(lsof -t -i:8000)`

### Decoder returns empty translations
The model isn't predicting EOS tokens. See "Loss stays at ~11.0" above.

### Coordinator can't find decoders
Redis is not running or decoders aren't registered.
**Fix:** `./uts serve --redis start`
