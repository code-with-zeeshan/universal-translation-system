# UniversalTranslationSDK for Flutter

A Flutter SDK for the Universal Translation System with coordinator-aware routing and auto-updating encoder.

## Features
- **Edge encoding** via FFI (privacy-preserving)
- **Coordinator-aware**: single decoder → direct, multiple → proxy through coordinator
- **Auto-updating encoder**: checks HF Hub for newer model at init
- **Dynamic vocabulary packs** from HF Hub with CDN fallback

## Quick Start

```dart
import 'package:universal_translation_sdk/universal_translation_sdk.dart';

// Direct decoder
final client = TranslationClient(decoderUrl: 'http://decoder:8000');

// With coordinator (auto-routes based on pool size)
final client = TranslationClient(
  decoderUrl: 'http://decoder:8000',
  coordinatorUrl: 'http://coordinator:5100',
);

await client.initialize();
final result = await client.translate(text: 'Hello world', from: 'en', to: 'es');
```

## Constructor

```dart
TranslationClient({
  String decoderUrl = 'https://api.yourdomain.com/decode',
  String? coordinatorUrl,
  Duration timeout = Duration(seconds: 30),
})
```

## Documentation
- [SDK Integration](../../docs/SDK_INTEGRATION.md)
- [Architecture](../../docs/ARCHITECTURE.md)
- [Publishing](../../docs/SDK_PUBLISHING.md)
