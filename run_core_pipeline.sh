#!/usr/bin/env bash
set -euo pipefail

# ── Run the core data pipeline ──────────────────────────────────
# Skips wikipedia_backtranslation, direct_opus, knowledge_distillation
# Runs: download → sample → augment (false-friends + idioms + backtranslation)
#        → create_ready → comet_filter → validate → vocabulary

export UTS_HMAC_KEY="dev-only-change-in-production-1234567890abc"

cd "$(dirname "$0")"

echo "=== Pipeline: core stages only ==="
echo "Stages: download_evaluation, download_training, sample_filter,"
echo "        augment, create_ready, comet_quality, validate, vocabulary"
echo ""

python -m data.unified_data_pipeline --config config/base.yaml --resume

echo ""
echo "=== Pipeline complete ==="
echo "Next: python -m training.intelligent_trainer --config config/base.yaml"
