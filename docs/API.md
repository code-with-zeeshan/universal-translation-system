# API Documentation

## Encoder API

### Python SDK

```python
from universal_translation import TranslationClient

client = TranslationClient(decoder_url="https://api.example.com/decode")

# Simple translation
result = await client.translate(
    text="Hello world",
    source_lang="en",
    target_lang="es"
)

# Batch translation
results = await client.translate_batch([
    {"text": "Hello", "source": "en", "target": "es"},
    {"text": "World", "source": "en", "target": "fr"}
])
```

### Flutter SDK
```dart
import 'package:universal_translation_sdk/universal_translation_sdk.dart';
final encoder = TranslationEncoder();
await encoder.initialize();
await encoder.loadVocabulary('en', 'es');
final encoded = await encoder.encode(text: 'Hello, world!', sourceLang: 'en', targetLang: 'es');
// Send encoded data to cloud decoder
```

### REST API (Decoder, served by Litserve)

#### POST /decode
- **Headers**:
  - `Content-Type`: `application/octet-stream`
  - `X-Target-Language`: `[language_code]`
- **Body**: Binary compressed encoder output
- **Response**: `application/json`
```json
{
  "translation": "Hola mundo",
  "target_lang": "es",
  "confidence": 0.95
}
```

#### GET /health
- Returns 200 if service is healthy

#### GET /metrics
- Prometheus metrics for monitoring

### Language Codes
|**Language** |	**Code** |
|-------------|----------|
|English	   |       en|
|Spanish	   |       es|
|French	       |  fr|
|German	       | de|
|Chinese	   |    zh|
|Japanese	   | ja|
|Korean	      |    ko|
|Arabic	      |    ar|
|Hindi	      |    hi|
|Russian	  |         ru|
|Portuguese	  |    pt|
|Italian	   |       it|
|Turkish	  |        tr|
|Thai	      |    th|
|Vietnamese	  |    vi|
|Polish	      |    pl|
|Ukrainian	  |    uk|
|Dutch	      |    nl|
|Indonesian	 |     id|
|Swedish	  |        sv|

## SDK Methods

### Android/Kotlin
```java
class TranslationClient(context: Context) {
    suspend fun translate(
        text: String,
        sourceLang: String,
        targetLang: String
    ): TranslationResult
    
    suspend fun prepareTranslation(
        sourceLang: String,
        targetLang: String
    ): Boolean
    
    fun getSupportedLanguages(): List<String>
    
    fun getDownloadedVocabularies(): List<VocabularyInfo>
}
```

### iOS/Swift
```swift
class TranslationClient {
    func translate(
        text: String,
        from sourceLang: String,
        to targetLang: String
    ) async throws -> String
    
    func prepareTranslation(
        sourceLang: String,
        targetLang: String
    ) async throws
    
    func supportedLanguages() -> [String]
}
```

### JavaScript/TypeScript
```javascript
interface TranslationOptions {
    text: string;
    sourceLang: string;
    targetLang: string;
}

class TranslationClient {
    async translate(options: TranslationOptions): Promise<string>
    async prepareTranslation(source: string, target: string): Promise<void>
    getSupportedLanguages(): string[]
}
```

**Error Codes** 

|Code	   |      Description |
|----------|-----------|
|400	   |           Invalid language code |
|401	   |           Authentication required |
|413	   |           Text too long (>1000 chars) |
|422	   |           Unsupported language pair |
|429	   |           Rate limit exceeded |
|500	   |           Server error |
|503	   |           Service unavailable |