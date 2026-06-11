# Vocabulary Pack Guide

## Overview

The Universal Translation System uses a dynamic vocabulary system for efficient language support. Vocabulary packs are small (2-4MB each), language-specific, and can be downloaded on-demand, reducing overall app size while maintaining translation quality.

**Quick command:** `./uts vocab --help`

## Vocabulary System Architecture

### Core Components

1. **Universal Encoder (edge-optimized)**
   - Language-agnostic encoder working with any vocabulary pack
   - Optimized for mobile and web deployment via quantization
   - Dynamically resizes embedding table to match loaded pack

2. **Vocabulary Packs (2-4MB each, 32K tokens)**
   - Latin Pack: English, Spanish, French, German, Italian, Portuguese, Dutch, Swedish, Polish, Indonesian, Vietnamese, Turkish
   - CJK Pack: Chinese, Japanese, Korean
   - Arabic Pack: Arabic
   - Devanagari Pack: Hindi
   - Cyrillic Pack: Russian, Ukrainian
   - Thai Pack: Thai

3. **Dynamic Loading System**
   - Packs loaded only when needed
   - Memory-efficient with LRU caching
   - Versioned via semver in filename (`latin_v1.0.0.msgpack`)

### Implementation Modules

The vocabulary system is split into modular components under `pipeline/vocabulary/` and `runtime/vocabulary/`:
- `pipeline/vocabulary/creator.py` -- `UnifiedVocabularyCreator` (main entry point)
- `pipeline/vocabulary/production.py` -- SentencePiece-based production vocabulary creation
- `pipeline/vocabulary/research.py` -- Frequency-based research/alternative creation
- `pipeline/vocabulary/validation.py` -- Pack validation utilities
- `pipeline/vocabulary/config.py` -- `CreationMode`, `UnifiedVocabConfig`, `VocabStats`
- `runtime/vocabulary/manager.py` -- Runtime vocabulary management
- `pipeline/vocabulary/evolve.py` -- Promotes unknown tokens and retrains model embeddings
- `tools/vocab_adapter.py` -- `EmbeddingResizeAdapter` for resizing encoder/decoder embeddings during evolution

## Using Vocabulary Packs

### In SDKs
```javascript
// Web/React Native SDK
const translator = new TranslationClient();
const result = await translator.translate({
  text: "Hello",
  sourceLang: "en",
  targetLang: "zh" // Downloads CJK pack if not cached
});
```

```swift
// iOS SDK
let translator = TranslationClient()
let result = try await translator.translate(text: "Hello", from: "en", to: "zh")
```

### From Data Pipeline
- After data is processed, the pipeline triggers vocabulary creation.
- See `pipeline/connectors/vocabulary.py` and `pipeline/vocabulary/creator.py`.
- Runtime configuration:
   - `vocabulary.vocab_dir`: base directory for packs (default `vocabulary/vocab`, overridable via `UTS_VOCABS_DIR`)
  - `vocabulary.language_to_pack_mapping`: e.g., `en,es,fr,de -> latin`; `zh,ja,ko -> cjk`

### Creating Custom Vocabulary Packs
```bash
# Via unified CLI
./uts vocab --build --vocab-size 32000 --mode production

# Programmatic
python -c "
from pipeline.vocabulary.creator import UnifiedVocabularyCreator, CreationMode
creator = UnifiedVocabularyCreator(corpus_dir='data/processed', output_dir='vocabulary/vocab')
creator.create_pack(pack_name='medical', languages=['en','es','fr'], mode=CreationMode.PRODUCTION)
"

# Evolve vocabulary (promote unknown tokens + retrain model embeddings)
python -m pipeline.vocabulary.evolve --pack-name latin --config config/base.yaml --retrain-model --retrain-epochs 3
```

## Vocabulary Pack Features

- **Efficient Storage**: Small file size (2-4MB per pack)
- **Dynamic Loading**: Download only what you need
- **Memory Efficiency**: LRU caching for optimal memory usage
- **Bloom Filters**: Fast token lookup
- **Compression**: MsgPack-encoded, optimized for size and performance
- **Versioning**: Semantic versioning in filename (`_v{major}.{minor}.msgpack`)

## Best Practices

1. **Preload Common Languages**: Preload most common language packs for better UX
2. **Monitor Usage**: Use coordinator dashboard to track pack usage
3. **Version Control**: Track vocabulary pack versions for consistency
4. **Custom Domains**: Create domain-specific packs for specialized terminology

---

For more details, see [ARCHITECTURE.md](ARCHITECTURE.md) and [environment-variables.md](environment-variables.md).
