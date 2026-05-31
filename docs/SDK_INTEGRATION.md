# SDK Integration Guide for App Developers

This guide explains how to integrate the Universal Translation System SDKs into your mobile, web, and cross-platform applications. All SDKs live under the `sdk/` directory.

---

## Overview
- **Edge-native encoding** (where supported) for fast, private, on-device processing
- **Cloud-based decoding** for high-quality translations
- **Dynamic vocabulary management** for efficient language support
- **Coordinator routing** for load balancing via `/api/decode`

---

## Android Integration

### 1) Add the SDK
Refer to `sdk/android/UniversalTranslationSDK/`. Option A: Maven local, Option B: Direct module include, Option C: AAR.

### 2) Encode and POST
```kotlin
val encoder = TranslationEncoder(context)
encoder.loadVocabulary("en", "es")
val encoded: ByteArray = encoder.encode("Hello world", "en", "es")

// Coordinator (binary)
val request = okhttp3.Request.Builder()
    .url("https://coord.example.com/api/decode")
    .addHeader("Content-Type", "application/octet-stream")
    .addHeader("X-API-Key", YOUR_API_KEY)
    .addHeader("X-Source-Language", "en")
    .addHeader("X-Target-Language", "es")
    .post(okhttp3.RequestBody.create(null, encoded))
    .build()

val client = okhttp3.OkHttpClient()
client.newCall(request).execute().use { resp ->
    val body = resp.body?.string()
    // Parse JSON { translation: "..." }
}
```

---

## iOS Integration

### 1) Add the SDK
Refer to `sdk/ios/UniversalTranslationSDK/`. Use Swift Package Manager or CocoaPods.

### 2) Encode and POST
```swift
let encoder = TranslationEncoder()
try encoder.loadVocabulary(source: "en", target: "es")
let encoded: Data = try encoder.encode(text: "Hello world", source: "en", target: "es")

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
```

---

## Flutter Integration

### 1) Add the SDK
```yaml
# pubspec.yaml
universal_translation_sdk:
  path: ./sdk/flutter/universal_translation_sdk
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
```

---

## React Native Integration

### 1) Add the SDK
```bash
npm install @universal-translation/react-native-sdk
```

### 2) Usage Example
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

### 2) Usage Example
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

---

## Sample curl commands

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

## Headers for Binary Requests
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
- Vocabulary packs: `/app/vocabs`
- Client flow: Encode text -> POST to `/api/decode` (or `/decode`) -> Receive translation JSON

---

## Web WASM assets and hosting

Serve `dist/wasm/` with correct headers:
```nginx
location /wasm/ {
  alias /var/www/app/dist/wasm/;
  add_header Cross-Origin-Opener-Policy same-origin always;
  add_header Cross-Origin-Embedder-Policy require-corp always;
  add_header Access-Control-Allow-Origin *;
  types { application/wasm wasm; }
  expires 7d;
}
```

---

## Further Reading
- [ARCHITECTURE.md](./ARCHITECTURE.md)
- [API.md](./API.md)
- [DEPLOYMENT.md](./DEPLOYMENT.md)
- [CI_CD.md](./CI_CD.md)
