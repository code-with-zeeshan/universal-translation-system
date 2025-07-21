# Adding New Languages

To add a new language to the Universal Translation System, follow these steps:

## 1. Update Data Pipeline
- Add new language code to `data/config.yaml` under `languages:`
- Update `training_distribution` in `data/config.yaml` to include new language pairs

## 2. Download and Prepare Data
- Use `data/download_training_data.py` or `data/practical_data_pipeline.py` to fetch and preprocess parallel data for the new language
- Ensure data is placed in the correct directory structure (`data/raw`, `data/processed`)

## 3. Create/Update Vocabulary Packs
- Run `python vocabulary/create_vocabulary_packs_from_data.py` to generate or update vocabulary packs
- If the new language uses a unique script, create a new vocabulary group in `data/config.yaml` under `vocabulary_strategy.groups`

## 4. Update Model Training Configs
- Ensure the new language is included in all relevant `config/training_*.yaml` files
- Adjust batch size or training distribution as needed

## 5. Train or Fine-tune Models
- Use `scripts/train_from_scratch.sh` or `training/train_universal_system.py` to train or fine-tune the encoder/decoder
- Config auto-detection will select the best hardware config

## 6. Update SDKs
- Add the new language code to supported languages in each SDK (Android, iOS, Flutter, React Native, Web)
- Update language pickers and UI as needed

## 7. Register in Coordinator/Decoder Pool (if needed)
- If deploying a new decoder node for the language, use `tools/register_decoder_node.py` to add it to the pool
- Ensure the node passes health checks and is visible in the coordinator dashboard

## 8. Test End-to-End
- Use `tests/` and SDK integration tests to verify translation quality and system integration

## 9. Update Documentation
- Add the new language to all relevant docs and user guides

---

For more details, see [docs/SDK_INTEGRATION.md](../docs/SDK_INTEGRATION.md) and [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md).