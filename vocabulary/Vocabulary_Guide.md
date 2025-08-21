# Vocabulary Pack Guide

## 📋 Overview

The Universal Translation System uses a dynamic vocabulary system that allows for efficient language support. Vocabulary packs are small (2-4MB each), language-specific, and can be downloaded on-demand, reducing the overall app size while maintaining translation quality.

## 🔧 Vocabulary System Architecture

### Core Components

1. **Universal Encoder Base (35MB)**
   - Language-agnostic encoder that works with any vocabulary pack
   - Optimized for mobile and web deployment

2. **Vocabulary Packs (2-4MB each)**
   - Latin Pack (~3MB): Covers English, Spanish, French, German, Italian, Portuguese, etc.
   - CJK Pack (~4MB): Covers Chinese, Japanese, Korean
   - Other language-specific packs

3. **Dynamic Loading System**
   - Packs are loaded only when needed
   - Memory-efficient with LRU caching

## 🚀 Using Vocabulary Packs

### In SDKs

```javascript
// Web/React Native SDK example
const translator = new TranslationClient();

// The vocabulary pack will be automatically downloaded if not already available
const result = await translator.translate({
  text: "Hello",
  sourceLang: "en",
  targetLang: "zh" // Will download CJK pack if not already available
});
```

```swift
// iOS SDK example
let translator = TranslationClient()
let result = try await translator.translate(text: "Hello", from: "en", to: "zh")
```

### Creating Custom Vocabulary Packs

For domain-specific terminology or specialized use cases, you can create custom vocabulary packs:

1. Use `vocabulary/unified_vocabulary_creator.py` for creating new packs
2. Configure language settings in your environment variables
3. Register the pack with the coordinator for cloud decoding

## 📊 Vocabulary Pack Features

- **Efficient Storage**: Small file size (2-4MB per language group)
- **Dynamic Loading**: Download only what you need
- **Memory Efficiency**: LRU caching for optimal memory usage
- **Bloom Filters**: Fast token lookup
- **Compression**: Optimized for size and performance

## 💡 Best Practices

1. **Preload Common Languages**: For better user experience, preload the most common language packs
2. **Monitor Usage**: Use the coordinator dashboard to track vocabulary pack usage
3. **Version Control**: Keep track of vocabulary pack versions for consistency
4. **Custom Domains**: Consider creating domain-specific vocabulary packs for specialized terminology

## 🤝 Contributing

- Add support for new languages
- Improve compression techniques
- Enhance tokenization for specific languages
- Document your changes and update configurations as needed

---

For more details, see [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) and the environment variables documentation [docs/environment-variables.md](../docs/environment-variables.md).