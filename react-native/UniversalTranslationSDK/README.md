# Universal Translation SDK for React Native

React Native SDK for the Universal Translation System with on-device encoding and cloud decoding.

## Features

- 🚀 **On-device text encoding** for privacy and speed
- 🌍 **Support for 20+ languages** out of the box
- 📦 **Dynamic vocabulary loading** (2-8MB per language group)
- 💾 **Efficient caching** for offline support
- 📱 **iOS and Android support** with platform-specific optimizations
- ⚡ **TypeScript support** with full type definitions
- 🎣 **React hooks** for easy integration
- 🔒 **Privacy-first** architecture with local encoding

## Requirements

- React Native >= 0.72.0
- iOS >= 15.0
- Android >= 21 (API level 5.0)
- For iOS: Xcode >= 14.0
- For Android: Android Studio with NDK support

## Installation

```bash
npm install @universal-translation/react-native-sdk
# or
yarn add @universal-translation/react-native-sdk
```
### iOS Setup

```bash
cd ios && pod install
```

#### Additional iOS Configuration

Add to your `Info.plist`:
```xml
<key>NSAppTransportSecurity</key>
<dict>
    <key>NSAllowsArbitraryLoads</key>
    <false/>
    <key>NSExceptionDomains</key>
    <dict>
        <key>yourdomain.com</key>
        <dict>
            <key>NSExceptionAllowsInsecureHTTPLoads</key>
            <true/>
        </dict>
    </dict>
</dict>
```

### Android Setup

Add to your `android/app/build.gradle`:
```gradle
android {
    packagingOptions {
        pickFirst 'lib/x86/libc++_shared.so'
        pickFirst 'lib/x86_64/libc++_shared.so'
        pickFirst 'lib/arm64-v8a/libc++_shared.so'
        pickFirst 'lib/armeabi-v7a/libc++_shared.so'
    }
}
```

## Usage

### Basic Translation

```typescript
import { useTranslation } from '@universal-translation/react-native-sdk';

function MyComponent() {
  const { translate, isTranslating, error } = useTranslation({
    decoderUrl: 'https://api.yourdomain.com/decode' // Optional, uses default if not provided
  });

  const handleTranslate = async () => {
    try {
      const result = await translate({
        text: 'Hello world',
        sourceLang: 'en',
        targetLang: 'es',
      });
      console.log(result.translation); // "Hola mundo"
    } catch (err) {
      console.error('Translation failed:', err);
    }
  };

  return (
    <Button
      title="Translate"
      onPress={handleTranslate}
      disabled={isTranslating}
    />
  );
}
```

### Using the Translation Client

```typescript
import TranslationClient from '@universal-translation/react-native-sdk';

// Create client instance
const client = new TranslationClient({
  decoderUrl: 'https://api.yourdomain.com/decode',
  maxCacheSize: 100,
});

// Translate text
const result = await client.translate({
  text: 'Hello',
  sourceLang: 'en',
  targetLang: 'fr',
});

// Batch translation
const results = await client.translateBatch(
  ['Hello', 'World', 'How are you?'],
  'en',
  'es'
);

// Get supported languages
const languages = await client.getSupportedLanguages();

// Manage vocabularies
const vocabInfo = await client.getVocabularyInfo('en', 'es');
if (vocabInfo.needsDownload) {
  await client.downloadVocabulariesForLanguages(['en', 'es']);
}

// Clear cache
client.clearCache();
```

### Pre-built UI Component

```typescript
import { TranslationScreen } from '@universal-translation/react-native-sdk';

function App() {
  return (
    <TranslationScreen
      decoderUrl="https://api.yourdomain.com/decode"
      defaultSourceLang="en"
      defaultTargetLang="es"
    />
  );
}
```

### Advanced Usage with Progress Tracking

```typescript
import { useTranslation } from '@universal-translation/react-native-sdk';

function AdvancedTranslation() {
  const {
    translate,
    downloadLanguages,
    downloadProgress,
    isTranslating,
    error,
    clearError
  } = useTranslation();

  // Download languages with progress tracking
  const prepareLanguages = async () => {
    await downloadLanguages(['en', 'es', 'fr', 'de']);
  };

  // Monitor download progress
  useEffect(() => {
    if (downloadProgress.es) {
      console.log(`Spanish vocabulary: ${downloadProgress.es}% downloaded`);
    }
  }, [downloadProgress]);

  return (
    <View>
      {/* Your UI */}
    </View>
  );
}
```

## API Reference

### useTranslation Hook

```typescript
const {
  translate,           // Function to translate text
  translateBatch,      // Function to translate multiple texts
  isTranslating,       // Boolean indicating translation in progress
  error,               // Error message if translation fails
  clearError,          // Function to clear error state
  downloadProgress,    // Object with download progress for each language
  downloadLanguages,   // Function to download vocabulary packs
  getSupportedLanguages, // Function to get supported languages
  clearCache,          // Function to clear translation cache
} = useTranslation(options);
```

#### Options

- `decoderUrl` (string, optional): URL of the decoder service

### TranslationClient

```typescript
const client = new TranslationClient({
  decoderUrl?: string;    // Decoder service URL (default: https://api.yourdomain.com/decode)
  maxCacheSize?: number;  // Maximum cache entries (default: 100)
});
```

#### Methods

- `translate(options: TranslationOptions): Promise<TranslationResult>`
- `translateBatch(texts: string[], sourceLang: string, targetLang: string): Promise<TranslationResult[]>`
- `prepareLanguagePair(sourceLang: string, targetLang: string): Promise<void>`
- `getVocabularyInfo(sourceLang: string, targetLang: string): Promise<VocabularyPack>`
- `downloadVocabulariesForLanguages(languages: string[]): Promise<void>`
- `getSupportedLanguages(): Promise<LanguageInfo[]>`
- `getMemoryUsage(): Promise<number>`
- `clearCache(): void`
- `clearAllCaches(): Promise<void>`
- `subscribeToDownloadProgress(callback: (progress) => void): EmitterSubscription`

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

interface LanguageInfo {
  code: string;
  name: string;
  nativeName: string;
  isRTL: boolean;
}

interface VocabularyPack {
  name: string;
  languages: string[];
  downloadUrl: string;
  localPath: string;
  sizeMb: number;
  version: string;
  needsDownload: boolean;
}
```

## Supported Languages

| Code | Language | Native Name | Script |
|------|----------|-------------|--------|
| en | English | English | Latin |
| es | Spanish | Español | Latin |
| fr | French | Français | Latin |
| de | German | Deutsch | Latin |
| zh | Chinese | 中文 | Han |
| ja | Japanese | 日本語 | Kana/Kanji |
| ko | Korean | 한국어 | Hangul |
| ar | Arabic | العربية | Arabic |
| hi | Hindi | हिन्दी | Devanagari |
| ru | Russian | Русский | Cyrillic |
| pt | Portuguese | Português | Latin |
| it | Italian | Italiano | Latin |
| tr | Turkish | Türkçe | Latin |
| th | Thai | ไทย | Thai |
| vi | Vietnamese | Tiếng Việt | Latin |
| pl | Polish | Polski | Latin |
| uk | Ukrainian | Українська | Cyrillic |
| nl | Dutch | Nederlands | Latin |
| id | Indonesian | Bahasa Indonesia | Latin |
| sv | Swedish | Svenska | Latin |

## Performance Tips

1. **Pre-download vocabularies** for frequently used languages
2. **Cache translations** to reduce API calls
3. **Use batch translation** for multiple texts
4. **Clear cache periodically** to manage memory usage

## Troubleshooting

### iOS Build Issues

If you encounter build issues on iOS:
```bash
cd ios
pod deintegrate
pod install
```

### Android Build Issues

For Android build issues:
```bash
cd android
./gradlew clean
cd ..
npx react-native run-android
```

### Model Not Found

Ensure the model file is included in your app bundle:
- iOS: Add to Xcode project
- Android: Place in `android/app/src/main/assets/models/`

## Contributing

See the [contributing guide](CONTRIBUTING.md) to learn how to contribute to the repository.

## License

MIT

---

Made with ❤️ by the Universal Translation team
```