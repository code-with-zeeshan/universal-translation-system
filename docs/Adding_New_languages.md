# Adding New Languages

The system supports two scenarios:

## Scenario A: Adding to the initial 20-language set (before training)

If you haven't trained yet, add the language to config and train the full model:

```bash
# 1. Add language to config/base.yaml:
#    - data.active_languages
#    - data.active_pairs
#    - vocabulary.language_to_pack_mapping (assign to a script pack)
#    - vocabulary_strategy.groups (add to an existing pack or create new)

# 2. Run pipeline and train
./uts data --pipeline
./uts vocab --build
./uts train --full
```

## Scenario B: Adding language #21+ after backbone is trained

Freeze the trained backbone and train only LoRA adapters + target language adapter:

```yaml
# config/base.yaml
training:
  use_lora: true
  lora_r: 16
  lora_r_decoder: 64
```

```bash
# 1. Update config
# 2. Download data for new language(s)
./uts data --pipeline --stage download_training

# 3. Train adapters only (~2-3 hours)
./uts train --full --experiment-name "lang-21-adapter"
```

## Script Pack Assignment

| Script | Pack | Languages |
|---|---|---|
| Latin | `latin` | en, es, fr, de, it, pt, nl, sv, pl, id, vi, tr |
| CJK | `cjk` | zh, ja, ko |
| Arabic | `arabic` | ar |
| Devanagari | `devanagari` | hi |
| Cyrillic | `cyrillic` | ru, uk |
| Thai | `thai` | th |

If your new language uses a different script, create a new pack group in `config/base.yaml` → `vocabulary_strategy.groups`.

## Full Checklist

1. Add language code to `active_languages` in config
2. Assign to a script pack or create new pack
3. Add language pairs to `active_pairs`
4. Run pipeline to download and process data
5. Rebuild vocabulary packs
6. Train (full model or LoRA adapters)
7. Evaluate
