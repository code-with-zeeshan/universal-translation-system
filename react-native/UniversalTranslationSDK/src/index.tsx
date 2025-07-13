// react-native/UniversalTranslationSDK/src/index.tsx
import { NativeModules, Platform } from 'react-native';

const { UniversalTranslationModule } = NativeModules;

export interface TranslationOptions {
  text: string;
  sourceLang: string;
  targetLang: string;
}

export interface VocabularyInfo {
  name: string;
  languages: string[];
  sizeMB: number;
  isDownloaded: boolean;
}

class TranslationEncoder {
  private initialized: boolean = false;
  
  async initialize(): Promise<void> {
    if (!this.initialized) {
      await UniversalTranslationModule.initialize();
      this.initialized = true;
    }
  }
  
  async prepareTranslation(sourceLang: string, targetLang: string): Promise<void> {
    await this.initialize();
    return UniversalTranslationModule.prepareTranslation(sourceLang, targetLang);
  }
  
  async encode(text: string, sourceLang: string, targetLang: string): Promise<Uint8Array> {
    await this.initialize();
    const base64 = await UniversalTranslationModule.encode(text, sourceLang, targetLang);
    
    // Convert base64 to Uint8Array
    const binaryString = atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes;
  }
  
  async getAvailableVocabularies(): Promise<VocabularyInfo[]> {
    return UniversalTranslationModule.getAvailableVocabularies();
  }
  
  async downloadVocabulary(name: string): Promise<void> {
    return UniversalTranslationModule.downloadVocabulary(name);
  }
}

export class TranslationClient {
  private encoder: TranslationEncoder;
  private decoderUrl: string;
  
  constructor(decoderUrl: string = 'https://api.yourdomain.com/decode') {
    this.encoder = new TranslationEncoder();
    this.decoderUrl = decoderUrl;
  }
  
  async translate(options: TranslationOptions): Promise<string> {
    try {
      // Encode locally
      const encoded = await this.encoder.encode(
        options.text,
        options.sourceLang,
        options.targetLang
      );
      
      // Send to decoder
      const response = await fetch(this.decoderUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/octet-stream',
          'X-Target-Language': options.targetLang,
        },
        body: encoded,
      });
      
      if (!response.ok) {
        throw new Error(`Translation failed: ${response.status}`);
      }
      
      const result = await response.json();
      return result.translation;
    } catch (error) {
      throw new Error(`Translation error: ${error.message}`);
    }
  }
  
  async getVocabularyInfo(): Promise<VocabularyInfo[]> {
    return this.encoder.getAvailableVocabularies();
  }
  
  async downloadVocabulary(name: string): Promise<void> {
    return this.encoder.downloadVocabulary(name);
  }
}

// React Hook for easy usage
export function useTranslation() {
  const [client] = useState(() => new TranslationClient());
  const [isTranslating, setIsTranslating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const translate = useCallback(async (options: TranslationOptions) => {
    setIsTranslating(true);
    setError(null);
    
    try {
      const result = await client.translate(options);
      return result;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setIsTranslating(false);
    }
  }, [client]);
  
  return {
    translate,
    isTranslating,
    error,
    downloadVocabulary: client.downloadVocabulary.bind(client),
    getVocabularyInfo: client.getVocabularyInfo.bind(client),
  };
}

// Usage Example
export function TranslationScreen() {
  const { translate, isTranslating, error } = useTranslation();
  const [inputText, setInputText] = useState('');
  const [translatedText, setTranslatedText] = useState('');
  
  const handleTranslate = async () => {
    try {
      const result = await translate({
        text: inputText,
        sourceLang: 'en',
        targetLang: 'es',
      });
      setTranslatedText(result);
    } catch (err) {
      console.error('Translation failed:', err);
    }
  };
  
  return (
    <View style={styles.container}>
      <TextInput
        style={styles.input}
        value={inputText}
        onChangeText={setInputText}
        placeholder="Enter text to translate"
        multiline
      />
      
      <TouchableOpacity
        style={[styles.button, isTranslating && styles.buttonDisabled]}
        onPress={handleTranslate}
        disabled={isTranslating}
      >
        <Text style={styles.buttonText}>
          {isTranslating ? 'Translating...' : 'Translate to Spanish'}
        </Text>
      </TouchableOpacity>
      
      {error && <Text style={styles.error}>{error}</Text>}
      
      {translatedText && (
        <View style={styles.resultContainer}>
          <Text style={styles.resultText}>{translatedText}</Text>
        </View>
      )}
    </View>
  );
}