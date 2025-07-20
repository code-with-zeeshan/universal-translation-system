# Universal Translation Web SDK

Web SDK for the Universal Translation System with on-device encoding using ONNX Runtime Web.

## Features

- 🚀 **On-device encoding** - Privacy-first translation with local text encoding
- 🌍 **20+ languages** - Support for major world languages
- ⚡ **WebGL & WASM** - Hardware-accelerated inference in the browser
- 📦 **< 10MB runtime** - Lightweight with dynamic vocabulary loading
- 🔒 **Secure** - Your text never leaves your device unencrypted
- 💾 **Smart caching** - Automatic translation caching for better performance
- 📱 **Responsive** - Works on desktop and mobile browsers

## Installation

```bash
npm install @universal-translation/web-sdk
# or
yarn add @universal-translation/web-sdk
```

## Quick Start

### Basic Usage

```javascript
import { TranslationClient } from '@universal-translation/web-sdk';

// Initialize the client
const client = new TranslationClient({
  decoderUrl: 'https://api.yourdomain.com/decode' // Your decoder endpoint
});

// Initialize the encoder (loads the model)
await client.initialize();

// Translate text
const result = await client.translate({
  text: 'Hello world',
  sourceLang: 'en',
  targetLang: 'es'
});

console.log(result.translation); // "Hola mundo"
```

### React Component

```jsx
import { TranslationComponent } from '@universal-translation/web-sdk/react';

function App() {
  return (
    <TranslationComponent 
      defaultSourceLang="en"
      defaultTargetLang="es"
      decoderUrl="https://api.yourdomain.com/decode"
      onTranslation={(result) => console.log('Translated:', result)}
    />
  );
}
```

### Advanced Usage

```javascript
import { TranslationClient } from '@universal-translation/web-sdk';

const client = new TranslationClient({
  modelUrl: '/models/universal_encoder.onnx', // Custom model path
  decoderUrl: 'https://api.yourdomain.com/decode',
  maxCacheSize: 100, // Number of translations to cache
  headers: {
    'Authorization': 'Bearer YOUR_API_KEY' // Custom headers for decoder
  }
});

// Batch translation
const texts = ['Hello', 'How are you?', 'Goodbye'];
const results = await client.translateBatch(texts, 'en', 'es');

// Clear cache when needed
client.clearCache();

// Get supported languages
const languages = client.getSupportedLanguages();
// ['en', 'es', 'fr', 'de', 'zh', 'ja', 'ko', 'ar', 'hi', 'ru', ...]
```

## Setup Requirements

### 1. Model File

Place your ONNX model in your public directory:
```
public/
  models/
    universal_encoder.onnx
```

### 2. Vocabulary Packs

Convert and place vocabulary packs:
```
public/
  vocabs/
    latin_v1.0.json
    cjk_v1.0.json
    arabic_v1.0.json
    ...
```

### 3. WASM Files

The SDK will automatically load ONNX Runtime WASM files from `/wasm/`. Ensure they're served with proper MIME types:

```nginx
location ~ \.wasm$ {
    add_header Content-Type application/wasm;
}
```

### 4. CORS Configuration

Your decoder API must allow CORS requests:
```javascript
// Express example
app.use(cors({
  origin: ['https://yourdomain.com'],
  methods: ['POST'],
  allowedHeaders: ['Content-Type', 'X-Target-Language', 'X-Source-Language']
}));
```

## Converting Assets

### Model Conversion

```bash
# Install the SDK
npm install @universal-translation/web-sdk

# Run the model conversion guide
npx universal-translation-convert-model

# This will show you how to convert your PyTorch model to ONNX
```

### Vocabulary Conversion

```bash
# Convert MessagePack vocabularies to JSON
npx universal-translation-convert-vocab

# Or manually:
node node_modules/@universal-translation/web-sdk/scripts/convert-vocab.js
```

## API Reference

### TranslationClient

```typescript
const client = new TranslationClient(options?: {
  modelUrl?: string;      // Path to ONNX model (default: '/models/universal_encoder.onnx')
  decoderUrl?: string;    // Decoder API endpoint
  maxCacheSize?: number;  // Max cached translations (default: 100)
  headers?: Record<string, string>; // Additional headers for decoder requests
});
```

#### Methods

- `initialize(): Promise<void>` - Initialize the encoder model
- `translate(options: TranslationOptions): Promise<TranslationResult>` - Translate text
- `translateBatch(texts: string[], sourceLang: string, targetLang: string): Promise<TranslationResult[]>` - Translate multiple texts
- `clearCache(): void` - Clear translation cache
- `getSupportedLanguages(): string[]` - Get list of supported language codes

### Types

```typescript
interface TranslationOptions {
  text: string;
  sourceLang: string;
  targetLang: string;
}

interface TranslationResult {
  translation: string;
  targetLang: string;
  confidence?: number;
}
```

## Supported Languages

| Code | Language | Script |
|------|----------|--------|
| en | English | Latin |
| es | Spanish | Latin |
| fr | French | Latin |
| de | German | Latin |
| it | Italian | Latin |
| pt | Portuguese | Latin |
| nl | Dutch | Latin |
| sv | Swedish | Latin |
| pl | Polish | Latin |
| tr | Turkish | Latin |
| id | Indonesian | Latin |
| vi | Vietnamese | Latin |
| zh | Chinese | Han |
| ja | Japanese | Kana/Kanji |
| ko | Korean | Hangul |
| ar | Arabic | Arabic |
| hi | Hindi | Devanagari |
| ru | Russian | Cyrillic |
| uk | Ukrainian | Cyrillic |
| th | Thai | Thai |

## Browser Support

- Chrome 90+
- Firefox 89+
- Safari 15.4+
- Edge 90+

WebGPU support (experimental):
- Chrome 113+ with flag enabled

## Performance Tips

1. **Preload the model** - Call `initialize()` early
2. **Use caching** - Translations are automatically cached
3. **Batch requests** - Use `translateBatch()` for multiple texts
4. **Enable WebGL** - Ensure hardware acceleration is enabled
5. **Optimize vocabulary** - Only load language packs you need

## Troubleshooting

### Model Loading Issues

```javascript
// Enable debug logging
import * as ort from 'onnxruntime-web';
ort.env.logLevel = 'verbose';
```

### CORS Errors

Ensure your decoder API includes proper CORS headers:
```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: POST
Access-Control-Allow-Headers: Content-Type, X-Target-Language, X-Source-Language
```

### WebGL Not Available

The SDK will automatically fall back to WASM if WebGL is not available. To force WASM:

```javascript
const encoder = new TranslationEncoder({
  executionProviders: ['wasm'] // Exclude 'webgl'
});
```

## Development

```bash
# Clone the repo
git clone https://github.com/yourusername/universal-translation-system.git
cd web/universal-translation-sdk

# Install dependencies
npm install

# Run tests
npm test

# Build the SDK
npm run build

# Start dev server
npm run dev
```

## License

MIT

## Contributing

See [CONTRIBUTING.md](https://github.com/yourusername/universal-translation-system/blob/main/CONTRIBUTING.md)