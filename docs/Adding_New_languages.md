# Adding New Languages

To add a new language to the Universal Translation System, follow these steps:

## 1. Update Configuration
- Add new language code to `config/base.yaml` and/or environment variables.
- Update language pair configurations.

## 2. Download and Prepare Data
```bash
python scripts/pipeline.py data --config ./config/base.yaml --stages download_training create_ready
```

## 3. Create/Update Vocabulary Packs
```bash
python scripts/pipeline.py vocab --mode production --corpus-dir ./data/processed --output-dir vocabulary/vocab
```
- Programmatic alternative:
```python
from vocabulary.vocabulary_creator import UnifiedVocabularyCreator, CreationMode
creator = UnifiedVocabularyCreator(corpus_dir='data/processed', output_dir='vocabulary/vocab')
creator.create_pack(pack_name='latin', languages=['en','es','fr','de','new_lang'], mode=CreationMode.PRODUCTION)
```

## 4. Update Model Training Configs
- Ensure new language is included in training configuration.
- Adjust batch size or distribution as needed.

## 5. Train or Fine-tune Models
```bash
python -m training.launch train --config config/base.yaml
```

## 6. Update SDKs
- Add the new language code to supported languages in each SDK under `sdk/`.
- Update language pickers and UI.

## 7. Register in Coordinator/Decoder Pool
```bash
bash scripts/setup_serving.sh --register --endpoint https://your-decoder.com
```

## 8. Test End-to-End
```bash
pytest tests/test_translation.py -k "<new_language_code>"
```

## 9. Update Documentation
- Add the new language to all relevant docs.

## 10. Coordinator and Redis Notes
- If using Redis, the coordinator keeps a shared decoder pool.
- Pool is mirrored to `config/decoder_pool.json` periodically (`COORDINATOR_MIRROR_INTERVAL`).
- See `scripts/setup_redis.sh` for Redis setup.

---

For more details, see [SDK_INTEGRATION.md](SDK_INTEGRATION.md), [REDIS_INTEGRATION.md](REDIS_INTEGRATION.md), and [ARCHITECTURE.md](ARCHITECTURE.md).
