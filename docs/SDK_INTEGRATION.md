# SDK Integration Guide for App Developers

This guide explains how to integrate the Universal Translation System SDKs into your mobile, web, and cross-platform applications. It covers Android, iOS, Flutter, React Native, and Web.

---

## Overview
- **Edge-native encoding** (where supported) for fast, private, on-device processing
- **Cloud-based decoding** for high-quality, up-to-date translations
- **Dynamic vocabulary management** for efficient language support
- **Coordinator routing** for load balancing via `/api/decode`

---

## Android Integration

### 1) Add the SDK
- Use the native SDK (JNI bridge to C++ encoder core). Refer to `android/` and `react-native/UniversalTranslationSDK/android` for patterns.

### 2) Encode and POST (Coordinator and Decoder examples)
```kotlin
// build.gradle: add OkHttp
// implementation("com.squareup.okhttp3:okhttp:4.12.0")

val encoder = TranslationEncoder(context)
encoder.loadVocabulary("en", "es")
val encoded: ByteArray = encoder.encode("Hello world", "en", "es")

// Coordinator (binary)
val request1 = okhttp3.Request.Builder()
    .url("https://coord.example.com/api/decode")
    .addHeader("Content-Type", "application/octet-stream")
    .addHeader("X-API-Key", YOUR_API_KEY)
    .addHeader("X-Source-Language", "en")
    .addHeader("X-Target-Language", "es")
    .post(okhttp3.RequestBody.create(null, encoded))
    .build()

// Decoder (direct)
val request2 = okhttp3.Request.Builder()
    .url("https://decoder.example.com/decode")
    .addHeader("Content-Type", "application/octet-stream")
    .addHeader("X-Target-Language", "es")
    .post(okhttp3.RequestBody.create(null, encoded))
    .build()

val client = okhttp3.OkHttpClient()
client.newCall(request1).execute().use { resp ->
    val body = resp.body?.string()
    // Parse JSON { translation: "..." }
}
```

---

## iOS Integration

### 1) Add the SDK
- Use Swift Package Manager/CocoaPods. Refer to `ios/UniversalTranslationSDK`.

### 2) Encode and POST (Coordinator and Decoder)
```swift
let encoder = TranslationEncoder()
try encoder.loadVocabulary(source: "en", target: "es")
let encoded: Data = try encoder.encode(text: "Hello world", source: "en", target: "es")

// Coordinator
var req = URLRequest(url: URL(string: "https://coord.example.com/api/decode")!)
req.httpMethod = "POST"
req.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
req.setValue("YOUR_API_KEY", forHTTPHeaderField: "X-API-Key")
req.setValue("en", forHTTPHeaderField: "X-Source-Language")
req.setValue("es", forHTTPHeaderField: "X-Target-Language")
req.httpBody = encoded

URLSession.shared.dataTask(with: req) { data, res, err in
    guard let data = data else { return }
    // Parse JSON
}.resume()

// Decoder (direct)
var req2 = URLRequest(url: URL(string: "https://decoder.example.com/decode")!)
req2.httpMethod = "POST"
req2.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
req2.setValue("es", forHTTPHeaderField: "X-Target-Language")
req2.httpBody = encoded
```

---

## Flutter Integration

### 1) Add the SDK
```yaml
# pubspec.yaml
universal_translation_sdk:
  path: ./flutter/universal_translation_sdk
http: ^1.1.0
```

### 2) Encode and POST
```dart
import 'package:http/http.dart' as http;
import 'package:universal_translation_sdk/universal_translation_sdk.dart';

final encoder = TranslationEncoder();
await encoder.initialize();
await encoder.loadVocabulary('en', 'es');
final bytes = await encoder.encode(text: 'Hello, world!', sourceLang: 'en', targetLang: 'es');

// Coordinator
final resp = await http.post(
  Uri.parse('https://coord.example.com/api/decode'),
  headers: {
    'Content-Type': 'application/octet-stream',
    'X-API-Key': 'YOUR_API_KEY',
    'X-Source-Language': 'en',
    'X-Target-Language': 'es',
  },
  body: bytes,
);

// Decoder
final resp2 = await http.post(
  Uri.parse('https://decoder.example.com/decode'),
  headers: {
    'Content-Type': 'application/octet-stream',
    'X-Target-Language': 'es',
  },
  body: bytes,
);
```

---

## React Native Integration

### 1) Add the SDK
```bash
npm install @universal-translation/react-native-sdk
# or
yarn add @universal-translation/react-native-sdk
```

### 2) Usage Example (client handles encoding + binary POST)
```tsx
import { TranslationClient } from '@universal-translation/react-native-sdk';

const client = new TranslationClient({
  decoderUrl: 'https://coord.example.com/api/decode',
  timeout: 30000,
  retryCount: 2,
  apiKey: 'YOUR_API_KEY',
});

const result = await client.translate({ text: 'Hello world', sourceLang: 'en', targetLang: 'es' });
if (result.success) console.log(result.data.translation);
```

---

## Web Integration

### 1) Add the SDK
```bash
npm install @universal-translation/web-sdk
```

### 2) Usage Example (SDK)
```ts
import { TranslationClient } from '@universal-translation/web-sdk';

const client = new TranslationClient({
  decoderUrl: 'https://coord.example.com/api/decode',
  timeout: 30000,
  retryCount: 2,
  useWasm: true,
  apiKey: 'YOUR_API_KEY',
});

const result = await client.translate({ text: 'Hello world', sourceLang: 'en', targetLang: 'es' });
if (result.success) console.log(result.data.translation);
```

### 3) Manual fetch (WASM-encoded → binary POST)
```ts
// Assume you have a Uint8Array `encoded` from your WASM encoder
const res = await fetch('https://coord.example.com/api/decode', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/octet-stream',
    'X-API-Key': 'YOUR_API_KEY',
    'X-Source-Language': 'en',
    'X-Target-Language': 'es',
  },
  body: encoded, // Uint8Array is accepted by fetch
});
const data = await res.json();
```

---

## Sample curl commands (both endpoints)

### Coordinator (binary)
```bash
curl -X POST \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/octet-stream" \
  -H "X-Source-Language: en" \
  -H "X-Target-Language: es" \
  --data-binary @compressed_embeddings.bin \
  https://coord.example.com/api/decode
```

### Decoder (direct)
```bash
curl -X POST \
  -H "Content-Type: application/octet-stream" \
  -H "X-Target-Language: es" \
  --data-binary @compressed_embeddings.bin \
  https://decoder.example.com/decode
```

---

## Headers for Binary Requests (when manually calling HTTP)
- `Content-Type: application/octet-stream`
- `X-Source-Language: <code>` (coordinator)
- `X-Target-Language: <code>`
- `X-Domain: <domain>` (optional)
- `X-API-Key: <key>` (coordinator)

---

## Best Practices
- Preload vocabularies for frequently used pairs
- Handle retries and backoff; respect 429 rate limits
- Keep encoding on-device for privacy when possible
- Batch on server for throughput; use coordinator for load balancing

---

## Troubleshooting
- See [TROUBLESHOOT.md](./TROUBLESHOOT.md)
- Check platform-specific READMEs under each SDK folder

---

## Using Trained Artifacts
- Decoder expects: `/app/models/production/decoder.pt`
- Vocabulary packs mounted to: `/app/vocabs`
- Client flow:
  1. Encode text (WASM/native/JS)
  2. POST embeddings to `/api/decode` (or `/decode`) with headers
  3. Receive translation JSON

---

## Platform setup notes

### Android (Gradle)
```gradle
// app/build.gradle
android {
  defaultConfig {
    minSdkVersion 23 // or higher per your app
  }
  // If your project pins NDK, align with your CI image
  ndkVersion "25.2.9519653"
}

dependencies {
  implementation "com.squareup.okhttp3:okhttp:4.12.0"
  // Add your encoder core artifact or AAR if distributed internally
}
```

### iOS (SPM/Podfile)
- Swift Package Manager: Add your UniversalTranslationSDK package (or local path) in Xcode > Package Dependencies.
- CocoaPods Podfile example:
```ruby
platform :ios, '13.0'
use_frameworks!

target 'YourApp' do
  # If consuming via local path
  pod 'UniversalTranslationSDK', :path => '../ios/UniversalTranslationSDK'
end
```

- App Transport Security: ensure your endpoints are HTTPS in production; configure ATS exceptions only for local dev.

### React Native
- Ensure Hermes/JSI configuration is compatible if using native encoder bridges.
- iOS: run `pod install` in the example/ios or your app’s ios folder after adding the package.

## Web WASM assets and hosting

- Build copies all ONNX Runtime Web .wasm assets to `dist/wasm/` (see package.json copy-wasm script).
- Serve `dist/wasm/` with correct headers:

Nginx example:
```nginx
location /wasm/ {
  alias /var/www/app/dist/wasm/;
  add_header Cross-Origin-Opener-Policy same-origin always;
  add_header Cross-Origin-Embedder-Policy require-corp always;
  add_header Access-Control-Allow-Origin *; # restrict in prod
  types { application/wasm wasm; }
  expires 7d;
}
```

Express example:
```ts
import express from 'express';
import path from 'path';
const app = express();
const wasmDir = path.join(process.cwd(), 'dist', 'wasm');
app.use('/wasm', express.static(wasmDir, {
  setHeaders: (res) => {
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    res.setHeader('Access-Control-Allow-Origin', '*'); // restrict in prod
    res.setHeader('Cache-Control', 'public, max-age=604800, immutable');
  }
}));
```

Notes:
- Ensure all `.wasm` files from `onnxruntime-web/dist` are available; the SDK’s build script already copies `*.wasm` into `dist/wasm/`. Common files include:
  - ort-wasm.wasm
  - ort-wasm-simd.wasm
  - ort-wasm-threaded.wasm
- If using COOP/COEP headers, load your app under same origin and avoid mixed content.
- If you serve from a CDN, configure `application/wasm` content-type and caching.

---

## Further Reading
- [ARCHITECTURE.md](./ARCHITECTURE.md)
- [API.md](./API.md)
- [DEPLOYMENT.md](./DEPLOYMENT.md)
- [CI_CD.md](./CI_CD.md)