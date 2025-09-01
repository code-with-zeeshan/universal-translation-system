# Adding New Languages

To add a new language to the Universal Translation System, follow these steps:

## 1. Update Environment Configuration
- Add new language code to your environment variables or `.env` file
- Update language pair configurations to include the new language

## 2. Download and Prepare Data
- Use the unified data downloader to fetch and preprocess parallel data for the new language
- Ensure data is placed in the correct directory structure (`data/raw`, `data/processed`)

```bash
# Example command to download data for a new language
python -m data.unified_data_downloader --language_code <new_language_code>
```

## 3. Create/Update Vocabulary Packs
- Run the unified vocabulary creator to generate or update vocabulary packs
- If the new language uses a unique script, create a new vocabulary group in your configuration

```bash
# Generate vocabulary pack for new language
python -m vocabulary.unified_vocabulary_creator --language_code <new_language_code>
```

## 4. Update Model Training Configs
- Ensure the new language is included in relevant training configuration files
- Adjust batch size or training distribution as needed

## 5. Train or Fine-tune Models
- Use the intelligent trainer to train or fine-tune the encoder/decoder
- The system will automatically select the best hardware configuration

```bash
# Train with the new language included
# Option 1: Archived GPU config preset
python -m training.launch train --config config/archived_gpu_configs/training_generic_gpu.yaml

# Option 2: Dynamic config (no YAML)
python -m training.launch train --config dynamic --dynamic
```

## 6. Update SDKs
- Add the new language code to supported languages in each SDK (Android, iOS, Flutter, React Native, Web)
- Update language pickers and UI as needed

## 7. Register in Coordinator/Decoder Pool
- If deploying a new decoder node for the language, register it with the coordinator
- Ensure the node passes health checks and is visible in the coordinator dashboard

```bash
# Register a decoder node supporting the new language
universal-decoder-node register --name "your-node-name" --languages "en,es,<new_language_code>" --endpoint "https://your-decoder.com"
```

## 8. Test End-to-End
- Use the test suite and SDK integration tests to verify translation quality and system integration

```bash
# Run tests for the new language
pytest tests/test_translation.py -k "<new_language_code>"
```

## 9. Update Documentation
- Add the new language to all relevant docs and user guides

## 10. Coordinator and Redis Notes
- If you use Redis (set `REDIS_URL`), the coordinator keeps a shared decoder pool.
- The pool is mirrored to `configs/decoder_pool.json` periodically.
- Control the mirroring interval with `COORDINATOR_MIRROR_INTERVAL` (default 60s, minimum enforced at 5s).

---

For more details, see [SDK_INTEGRATION.md](SDK_INTEGRATION.md), [REDIS_INTEGRATION.md](REDIS_INTEGRATION.md), and [ARCHITECTURE.md](ARCHITECTURE.md).