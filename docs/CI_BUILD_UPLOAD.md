# CI: Build and Upload Artifacts to Hugging Face

This repository provides a GitHub Actions workflow to build/convert artifacts, create vocabulary packs, and upload everything to the Hugging Face Hub sequentially.

## Workflow: build-upload.yml
- Location: `.github/workflows/build-upload.yml`
- Triggers:
  - Tag pushes matching `v*.*.*`
  - Manual dispatch with inputs (repo_id, create_vocabs, convert_models, vocab_groups, vocab_mode)

## Prerequisites
- A Hugging Face Hub repository for artifacts (models, vocabs, adapters)
- `HF_TOKEN` secret for authentication
- `HF_HUB_REPO_ID` secret for repository id (or pass as manual input)

## What it does
1. Sets up Python and installs dependencies
2. Optionally logs in to HF Hub if `HF_TOKEN` is present
3. Runs `scripts/build_and_upload_pipeline.py`:
   - `--create-vocabs` calls `pipeline/vocabulary/creator.py`
   - `--convert-models` calls `tools/convert.py`
   - Always uploads via `scripts/upload_artifacts.py`

## Usage

### A) On tag push
```bash
git tag v1.0.0
git push --tags
```

### B) Manually
- GitHub -> Actions -> Build and Upload Artifacts -> Run workflow
- Provide inputs: `repo_id`, `create_vocabs`, `convert_models`, `vocab_groups`, `vocab_mode`

## Local pipeline (optional)
```bash
python scripts/build_and_upload_pipeline.py \
  --repo-id code-with-zeeshan/universal-translation-system \
  --create-vocabs --convert-models \
  --vocab-groups latin cjk --vocab-mode production
```

## Decoder prefetch guidelines
Add to `.env` on decoder nodes:
```ini
HF_HUB_REPO_ID=code-with-zeeshan/universal-translation-system
PREFETCH_VOCAB_GROUPS=latin,cjk
PREFETCH_ADAPTERS=es,fr,ja
```
On startup the decoder will prefetch `models/production/decoder.pt` and specified packs/adapters.
