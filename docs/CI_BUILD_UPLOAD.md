# CI: Build and Upload Artifacts to Hugging Face

This repository provides a GitHub Actions workflow to build/convert artifacts, create vocabulary packs, and upload everything to the Hugging Face Hub sequentially.

## Workflow: build-upload.yml
- Location: `.github/workflows/build-upload.yml`
- Triggers:
  - Tag pushes matching `v*.*.*`
  - Manual dispatch with inputs (repo_id, create_vocabs, convert_models, vocab_groups, vocab_mode)

## Prerequisites
- A Hugging Face Hub repository for artifacts (models, vocabs, adapters)
- A Hugging Face access token stored as a repository secret:
  - `HF_TOKEN`: used by the workflow to authenticate and push artifacts
- Optionally store the repository id as a secret:
  - `HF_HUB_REPO_ID`, or pass it as a manual input when dispatching the workflow

## What it does
1. Sets up Python and installs dependencies (including `huggingface_hub`)
2. Optionally logs in to HF Hub if `HF_TOKEN` is present
3. Runs `scripts/build_and_upload_pipeline.py` with the provided options:
   - `--create-vocabs` (optional) calls `vocabulary/unified_vocabulary_creator.py`
   - `--convert-models` (optional) calls `training/convert_models.py`
   - Always uploads via `scripts/upload_artifacts.py`

## Usage

### A) On tag push
- Create a version tag locally and push:
  ```bash
  git tag v1.0.0
  git push --tags
  ```
- The workflow uses `${{ secrets.HF_TOKEN }}` and `${{ secrets.HF_HUB_REPO_ID }}` if set.

### B) Manually
- Go to GitHub → Actions → Build and Upload Artifacts → Run workflow
- Provide inputs:
  - `repo_id`: e.g. `your-username/universal-translation-system`
  - `create_vocabs`: true/false
  - `convert_models`: true/false
  - `vocab_groups`: space-separated list (e.g. `latin cjk`)
  - `vocab_mode`: `production` | `research` | `hybrid`

## Local pipeline (optional)
You can run the same pipeline locally:
```bash
python scripts/build_and_upload_pipeline.py \
  --repo_id your-username/universal-translation-system \
  --create-vocabs --convert-models \
  --vocab-groups latin cjk --vocab-mode production
```

## Decoder prefetch guidelines
Add to `.env` (or environment) on your decoder nodes:
```ini
HF_HUB_REPO_ID=your-username/universal-translation-system
# Optional prefetch hints (comma-separated)
PREFETCH_VOCAB_GROUPS=latin,cjk
PREFETCH_ADAPTERS=es,fr,ja
```
On startup the decoder will attempt to prefetch:
- `models/production/decoder.pt`
- Any specified packs and adapters
Additionally, per request batch it will ensure vocab/adapters for the requested language pairs exist locally (best-effort).