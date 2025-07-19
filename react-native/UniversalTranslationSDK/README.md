# Universal Translation SDK for React Native

React Native SDK for the Universal Translation System with on-device encoding and cloud decoding.

## Features

- 🚀 On-device text encoding for privacy
- 🌍 Support for 20+ languages
- 📦 Dynamic vocabulary loading
- 💾 Efficient caching
- 📱 iOS and Android support
- ⚡ TypeScript support
- 🎣 React hooks for easy integration

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

### Android Setup
No additional setup required

## Usage
### Basic Translation
```React
import { useTranslation } from '@universal-translation/react-native-sdk';

function MyComponent() {
  const { translate, isTranslating, error } = useTranslation();

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
```React
import TranslationClient from '@universal-translation/react-native-sdk';

const client = new TranslationClient({
  decoderUrl: 'https://api.yourdomain.com/decode',
  timeout: 30000,
});

// Translate text
const result = await client.translate({
  text: 'Hello',
  sourceLang: 'en',
  targetLang: 'fr',
});

// Get supported languages
const languages = await client.getSupportedLanguages();

// Manage vocabularies
const vocabs = await client.getVocabularyInfo();
await client.downloadVocabulary('latin');
```
### Pre-built UI Component
```React
import { TranslationScreen } from '@universal-translation/react-native-sdk';

function App() {
  return (
    <TranslationScreen
      decoderUrl="https://api.yourdomain.com/decode"
    />
  );
}
```
## API Reference
### useTranslation Hook
```React
const {
  translate,      // Function to translate text
  isTranslating,  // Boolean indicating translation in progress
  error,          // Error message if translation fails
  progress,       // Download progress (0-100)
  downloadVocabulary,    // Function to download vocabulary pack
  getVocabularyInfo,     // Function to get vocabulary information
  getSupportedLanguages, // Function to get supported languages
  clearCache,     // Function to clear translation cache
} = useTranslation(options);
```
### TranslationClient
```React
const client = new TranslationClient({
  decoderUrl?: string;    // Decoder service URL
  headers?: Record<string, string>; // Additional headers
  timeout?: number;       // Request timeout in ms
  maxCacheSize?: number;  // Maximum cache entries
});
```
## Supported Languages
- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Chinese (zh)
- Japanese (ja)
- Korean (ko)
- Arabic (ar)
- Hindi (hi)
- Russian (ru)
- Portuguese (pt)
- And more...

## License
MIT

"""

This complete React Native SDK includes:
- ✅ TypeScript support with proper types
- ✅ Native modules for iOS and Android
- ✅ React hooks for easy integration
- ✅ Pre-built UI components
- ✅ Error handling and retry logic
- ✅ Progress tracking for downloads
- ✅ Caching support
- ✅ Memory management
- ✅ Comprehensive documentation
- ✅ Build configuration for both platforms
"""