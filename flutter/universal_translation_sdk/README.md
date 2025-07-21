# Universal Translation SDK for Flutter

A cross-platform Flutter SDK for integrating the Universal Translation System into your mobile and desktop apps. This SDK provides native-speed encoding on the edge (using FFI to the C++ core) and seamless cloud-based decoding for high-quality, low-latency translation.

---

## Features
- **Edge-native encoding** using the universal C++ encoder core (via FFI)
- **Cloud-based decoding** via REST/gRPC API
- **Vocabulary management** and dynamic language support
- **Works on Android, iOS, macOS, Linux, Windows**
- **Easy integration** with your Flutter apps

---

## Installation

Add the package to your `pubspec.yaml`:

```yaml
# pubspec.yaml
universal_translation_sdk:
  path: ./universal_translation_sdk
```

Then run:
```sh
flutter pub get
```

---

## Platform Setup

### Android
- Ensure the native encoder library (`libuniversal_encoder.so`) is included in your app's `android/app/src/main/jniLibs/` directory.
- Update your `android/app/build.gradle` to package native libraries if needed.

### iOS
- The encoder core is linked via CocoaPods. Make sure your Podfile includes the correct references.
- Run `pod install` in the `ios/` directory after adding the SDK.

### Desktop (macOS, Windows, Linux)
- Place the appropriate native library (`.dylib`, `.dll`, `.so`) in your app's executable directory.

---

## Usage Example

```dart
import 'package:universal_translation_sdk/universal_translation_sdk.dart';

void main() async {
  final encoder = TranslationEncoder();
  await encoder.initialize();

  // Load vocabulary for a language pair
  await encoder.loadVocabulary('en', 'es');

  // Encode text on device
  final encoded = await encoder.encode(
    text: 'Hello, world!',
    sourceLang: 'en',
    targetLang: 'es',
  );

  // Send encoded data to cloud decoder (example)
  // final translation = await sendToCloudDecoder(encoded);
}
```

---

## Architecture
- **TranslationEncoder**: FFI wrapper for the native C++ encoder core.
- **VocabularyManager**: Handles loading and caching of vocabulary packs.
- **Cloud Decoder**: Expects encoded data from the edge, returns translated text.

---

## API Reference
- See the Dart docs in `lib/` and `lib/src/` for full API details.
- Main classes: `TranslationEncoder`, `VocabularyManager`, `VocabularyPack`.

---

## Contributing
Pull requests and issues are welcome! Please see the main repo's CONTRIBUTING.md for guidelines.

---

## License
This SDK is part of the Universal Translation System. See the main LICENSE file for details. 