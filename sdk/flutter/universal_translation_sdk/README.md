# UniversalTranslationSDK for Flutter

A cross-platform Flutter SDK for the Universal Translation System, supporting dynamic vocabulary packs and coordinator integration.

## Features
- Edge encoding, cloud decoding (privacy-preserving)
- Dynamic vocabulary packs
- Coordinator integration
- Cross-platform (Android, iOS, desktop)

## Quick Start

1) Add to `pubspec.yaml`:
```yaml
dependencies:
  universal_translation_sdk:
    path: ../../sdk/flutter/universal_translation_sdk
```

2) Initialize and use:
```dart
import 'package:universal_translation_sdk/universal_translation_sdk.dart';
import 'package:http/http.dart' as http;

final encoder = TranslationEncoder();
await encoder.initialize();
await encoder.loadVocabulary('en', 'es');
final encoded = await encoder.encode(text: 'Hello, world!', sourceLang: 'en', targetLang: 'es');

final resp = await http.post(
  Uri.parse('http://localhost:5100/api/decode'),
  headers: {
    'Content-Type': 'application/octet-stream',
    'X-Source-Language': 'en',
    'X-Target-Language': 'es',
  },
  body: encoded,
);
```

## Documentation
- See [docs/SDK_INTEGRATION.md](../../docs/SDK_INTEGRATION.md) and [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md)

---

For more, see the main repo.
