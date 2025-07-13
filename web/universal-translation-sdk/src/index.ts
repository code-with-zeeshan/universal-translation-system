// web/universal-translation-sdk/src/index.ts
import * as ort from 'onnxruntime-web';

export interface TranslationOptions {
  text: string;
  sourceLang: string;
  targetLang: string;
}

export interface VocabularyPack {
  name: string;
  languages: string[];
  tokens: Map<string, number>;
  subwords: Map<string, number>;
  sizeMB: number;
}

export class TranslationEncoder {
  private session: ort.InferenceSession | null = null;
  private vocabularyCache: Map<string, VocabularyPack> = new Map();
  private currentVocab: VocabularyPack | null = null;
  private modelUrl: string;
  
  constructor(modelUrl: string = '/models/universal_encoder.onnx') {
    this.modelUrl = modelUrl;
    
    // Configure ONNX Runtime for WebGL/WASM
    ort.env.wasm.wasmPaths = '/wasm/';
    ort.env.webgl.contextId = 'webgl2';
  }
  
  async initialize(): Promise<void> {
    if (this.session) return;
    
    try {
      // Load model
      this.session = await ort.InferenceSession.create(this.modelUrl, {
        executionProviders: ['webgl', 'wasm'],
        graphOptimizationLevel: 'all',
      });
      
      console.log('✅ Encoder model loaded');
    } catch (error) {
      throw new Error(`Failed to load encoder model: ${error}`);
    }
  }
  
  async prepareTranslation(sourceLang: string, targetLang: string): Promise<void> {
    await this.initialize();
    
    // Determine required vocabulary packs
    const requiredPacks = this.getRequiredPacks(sourceLang, targetLang);
    
    // Load vocabularies
    for (const packName of requiredPacks) {
      if (!this.vocabularyCache.has(packName)) {
        await this.loadVocabularyPack(packName);
      }
    }
    
    // Merge vocabularies
    this.currentVocab = this.mergeVocabularies(requiredPacks);
  }
  
  private async loadVocabularyPack(packName: string): Promise<void> {
    const response = await fetch(`/vocabularies/${packName}.json`);
    if (!response.ok) {
      throw new Error(`Failed to load vocabulary pack: ${packName}`);
    }
    
    const data = await response.json();
    const pack: VocabularyPack = {
      name: packName,
      languages: data.languages,
      tokens: new Map(Object.entries(data.tokens)),
      subwords: new Map(Object.entries(data.subwords)),
      sizeMB: data.sizeMB,
    };
    
    this.vocabularyCache.set(packName, pack);
  }
  
  private tokenize(text: string, sourceLang: string): Int32Array {
    if (!this.currentVocab) {
      throw new Error('No vocabulary loaded');
    }
    
    const tokens: number[] = [];
    
    // Add language token
    const langToken = `<${sourceLang}>`;
    if (this.currentVocab.tokens.has(langToken)) {
      tokens.push(this.currentVocab.tokens.get(langToken)!);
    }
    
    // Simple tokenization (production would use SentencePiece)
    const words = text.toLowerCase().split(/\s+/);
    
    for (const word of words) {
      if (this.currentVocab.tokens.has(word)) {
        tokens.push(this.currentVocab.tokens.get(word)!);
      } else {
        // Handle unknown words with subword tokenization
        const subwords = this.tokenizeUnknown(word);
        tokens.push(...subwords);
      }
    }
    
    // Add end token
    tokens.push(this.currentVocab.tokens.get('</s>') || 3);
    
    // Pad to 128
    while (tokens.length < 128) {
      tokens.push(this.currentVocab.tokens.get('<pad>') || 0);
    }
    
    return new Int32Array(tokens.slice(0, 128));
  }
  
  private tokenizeUnknown(word: string): number[] {
    if (!this.currentVocab) return [1]; // <unk>
    
    const tokens: number[] = [];
    
    // Try common prefixes
    const prefixes = ['un', 're', 'dis', 'pre', 'post'];
    for (const prefix of prefixes) {
      if (word.startsWith(prefix) && word.length > prefix.length) {
        if (this.currentVocab.tokens.has(prefix)) {
          tokens.push(this.currentVocab.tokens.get(prefix)!);
          const remaining = `##${word.slice(prefix.length)}`;
          if (this.currentVocab.subwords.has(remaining)) {
            tokens.push(this.currentVocab.subwords.get(remaining)!);
            return tokens;
          }
        }
      }
    }
    
    // Default to unknown
    return [this.currentVocab.tokens.get('<unk>') || 1];
  }
  
  async encode(text: string, sourceLang: string, targetLang: string): Promise<Uint8Array> {
    if (!this.session) {
      throw new Error('Encoder not initialized');
    }
    
    // Ensure vocabulary is loaded
    await this.prepareTranslation(sourceLang, targetLang);
    
    // Tokenize
    const tokens = this.tokenize(text, sourceLang);
    
    // Create tensor
    const inputTensor = new ort.Tensor('int32', tokens, [1, 128]);
    
    // Run inference
    const feeds = { input_ids: inputTensor };
    const results = await this.session.run(feeds);
    
    // Get output
    const output = results.encoder_output;
    const outputData = output.data as Float32Array;
    
    // Compress output
    return this.compressOutput(outputData, output.dims as number[]);
  }
  
  private compressOutput(data: Float32Array, shape: number[]): Uint8Array {
    // Quantize to Int8
    const maxVal = Math.max(...Array.from(data).map(Math.abs));
    const scale = maxVal > 0 ? 127 / maxVal : 1;
    
    const quantized = new Int8Array(data.length);
    for (let i = 0; i < data.length; i++) {
      quantized[i] = Math.round(data[i] * scale);
    }
    
    // Create output buffer with metadata
    const metadataSize = 12;
    const compressedData = pako.deflate(quantized);
    const output = new Uint8Array(metadataSize + compressedData.length);
    
    // Write metadata
    const view = new DataView(output.buffer);
    view.setInt32(0, shape[1], true);
    view.setInt32(4, shape[2], true);
    view.setFloat32(8, scale, true);
    
    // Write compressed data
    output.set(compressedData, metadataSize);
    
    return output;
  }
  
  private getRequiredPacks(sourceLang: string, targetLang: string): string[] {
    const packMapping: Record<string, string> = {
      'en': 'latin',
      'es': 'latin',
      'fr': 'latin',
      'de': 'latin',
      'zh': 'cjk',
      'ja': 'cjk',
      'ko': 'cjk',
      'ar': 'arabic',
      'hi': 'devanagari',
      'ru': 'cyrillic',
      'th': 'thai',
      // ... more mappings
    };
    
    const packs = new Set<string>();
    if (packMapping[sourceLang]) packs.add(packMapping[sourceLang]);
    if (packMapping[targetLang]) packs.add(packMapping[targetLang]);
    
    return Array.from(packs);
  }
  
  private mergeVocabularies(packNames: string[]): VocabularyPack {
    const merged: VocabularyPack = {
      name: `merged_${packNames.join('_')}`,
      languages: [],
      tokens: new Map(),
      subwords: new Map(),
      sizeMB: 0,
    };
    
    // Add special tokens
    merged.tokens.set('<pad>', 0);
    merged.tokens.set('<unk>', 1);
    merged.tokens.set('<s>', 2);
    merged.tokens.set('</s>', 3);
    
    let tokenId = 4;
    
    // Merge packs
    for (const packName of packNames) {
      const pack = this.vocabularyCache.get(packName);
      if (!pack) continue;
      
      merged.languages.push(...pack.languages);
      merged.sizeMB += pack.sizeMB;
      
      // Merge tokens
      for (const [token, _] of pack.tokens) {
        if (!merged.tokens.has(token)) {
          merged.tokens.set(token, tokenId++);
        }
      }
      
      // Merge subwords
      for (const [subword, _] of pack.subwords) {
        if (!merged.subwords.has(subword)) {
          merged.subwords.set(subword, tokenId++);
        }
      }
    }
    
    return merged;
  }
}

export class TranslationClient {
  private encoder: TranslationEncoder;
  private decoderUrl: string;
  
  constructor(options: {
    modelUrl?: string;
    decoderUrl?: string;
  } = {}) {
    this.encoder = new TranslationEncoder(options.modelUrl);
    this.decoderUrl = options.decoderUrl || 'https://api.yourdomain.com/decode';
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
}

// Usage example
async function translateExample() {
  const client = new TranslationClient();
  
  try {
    const translation = await client.translate({
      text: 'Hello, how are you?',
      sourceLang: 'en',
      targetLang: 'es',
    });
    
    console.log('Translation:', translation);
  } catch (error) {
    console.error('Translation failed:', error);
  }
}

// React component example
export function TranslationComponent() {
  const [client] = useState(() => new TranslationClient());
  const [inputText, setInputText] = useState('');
  const [translatedText, setTranslatedText] = useState('');
  const [isTranslating, setIsTranslating] = useState(false);
  
  const handleTranslate = async () => {
    setIsTranslating(true);
    
    try {
      const result = await client.translate({
        text: inputText,
        sourceLang: 'en',
        targetLang: 'es',
      });
      setTranslatedText(result);
    } catch (error) {
      console.error('Translation failed:', error);
      setTranslatedText('Translation failed');
    } finally {
      setIsTranslating(false);
    }
  };
  
  return (
    <div className="translation-container">
      <textarea
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
        placeholder="Enter text to translate"
        rows={4}
        cols={50}
      />
      
      <button
        onClick={handleTranslate}
        disabled={isTranslating || !inputText}
      >
        {isTranslating ? 'Translating...' : 'Translate to Spanish'}
      </button>
      
      {translatedText && (
        <div className="translation-result">
          <h3>Translation:</h3>
          <p>{translatedText}</p>
        </div>
      )}
    </div>
  );
}