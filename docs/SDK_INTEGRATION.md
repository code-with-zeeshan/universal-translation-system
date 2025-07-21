# SDK Integration Guide for App Developers

This guide explains how to integrate the Universal Translation System SDKs into your mobile, web, and cross-platform applications. It covers Android, iOS, Flutter, React Native, and Web.

---

## Overview
- **Edge-native encoding** (where supported) for fast, private, on-device processing
- **Cloud-based decoding** for high-quality, up-to-date translations
- **Dynamic vocabulary management** for efficient language support

---

## Android Integration

### 1. Add the SDK
- Add the UniversalTranslationSDK to your `build.gradle`:
  ```gradle
  dependencies {
      implementation 'com.universaltranslation:encoder-sdk:1.0.0'
  }
  ```
- Place the native encoder library (`libuniversal_encoder.so`) in `app/src/main/jniLibs/`.

### 2. Usage Example
```kotlin
val encoder = TranslationEncoder(context)
encoder.loadVocabulary("en", "es")
val encoded = encoder.encode("Hello world", "en", "es")
// Send encoded to cloud decoder via API
```

---

## iOS Integration

### 1. Add the SDK
- Add to your Podfile:
  ```ruby
  pod 'UniversalTranslationSDK', '~> 1.0'
  ```
- Run `pod install`.
- Add the native encoder library to your Xcode project if needed.

### 2. Usage Example
```swift
let encoder = TranslationEncoder()
try encoder.loadVocabulary(source: "en", target: "es")
let encoded = try encoder.encode(text: "Hello world", source: "en", target: "es")
// Send encoded to cloud decoder via API
```

---

## Flutter Integration

### 1. Add the SDK
- In your `pubspec.yaml`:
  ```yaml
  universal_translation_sdk:
    path: ./universal_translation_sdk
  ```
- Run `flutter pub get`.
- Place the native encoder library in the appropriate directory for your platform.

### 2. Usage Example
```dart
import 'package:universal_translation_sdk/universal_translation_sdk.dart';
final encoder = TranslationEncoder();
await encoder.initialize();
await encoder.loadVocabulary('en', 'es');
final encoded = await encoder.encode(text: 'Hello, world!', sourceLang: 'en', targetLang: 'es');
// Send encoded data to cloud decoder
```

---

## React Native Integration

### 1. Add the SDK
- Install via npm or yarn:
  ```bash
  npm install universal-translation-sdk
  # or
  yarn add universal-translation-sdk
  ```

### 2. Usage Example
```tsx
import { useTranslation } from 'universal-translation-sdk';
const { translate } = useTranslation({ decoderUrl: 'https://api.example.com/decode' });
const result = await translate({ text: 'Hello world', sourceLang: 'en', targetLang: 'es' });
```

---

## Web Integration

### 1. Add the SDK
- Install via npm:
  ```bash
  npm install universal-translation-sdk
  ```
- Or include via CDN:
  ```html
  <script src="https://cdn.example.com/universal-translation-sdk.min.js"></script>
  ```

### 2. Usage Example
```js
import { TranslationClient } from 'universal-translation-sdk';
const client = new TranslationClient({ decoderUrl: 'https://api.example.com/decode' });
const result = await client.translate({ text: 'Hello world', sourceLang: 'en', targetLang: 'es' });
```

---

## Best Practices
- Always check for supported languages before translating.
- Preload vocabularies for frequently used language pairs.
- Handle network errors and retries gracefully.
- For privacy, keep encoding on-device where possible.
- Use batching for high-throughput translation needs.

---

## Troubleshooting
- See [TROUBLESHOOT.md](./TROUBLESHOOT.md) for common issues.
- For platform-specific issues, consult the SDK README in each SDK folder.

---

## Further Reading
- [ARCHITECTURE.md](./ARCHITECTURE.md)
- [API.md](./API.md)
- [DEPLOYMENT.md](./DEPLOYMENT.md)
- [CI_CD.md](./CI_CD.md) 