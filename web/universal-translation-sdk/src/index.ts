// src/index.ts
import * as ort from 'onnxruntime-web';

// Types
export interface TranslationOptions {
  text: string;
  sourceLang: string;
  targetLang: string;
}

export interface TranslationResult {
  translation: string;
  targetLang: string;
  confidence?: number;
}

export interface VocabularyPack {
  name: string;
  version: string;
  languages: string[];
  tokens: Record<string, number>;
  subwords: Record<string, number>;
  special_tokens: Record<string, number>;
  metadata: {
    total_tokens: number;
    size_mb: number;
    coverage_percentage?: number;
    oov_rate?: number;
  };
}

export interface EncoderConfig {
  modelUrl?: string;
  vocabUrl?: string;
  wasmPaths?: string;
  executionProviders?: Array<'webgl' | 'wasm' | 'webgpu'>;
  graphOptimizationLevel?: 'disabled' | 'basic' | 'extended' | 'all';
}

// Language mapping matching your system
const LANGUAGE_TO_PACK: Record<string, string> = {
  'en': 'latin', 'es': 'latin', 'fr': 'latin', 'de': 'latin',
  'it': 'latin', 'pt': 'latin', 'nl': 'latin', 'sv': 'latin',
  'pl': 'latin', 'id': 'latin', 'vi': 'latin', 'tr': 'latin',
  'zh': 'cjk', 'ja': 'cjk', 'ko': 'cjk',
  'ar': 'arabic', 'hi': 'devanagari',
  'ru': 'cyrillic', 'uk': 'cyrillic',
  'th': 'thai'
};

export class TranslationEncoder {
  private session: ort.InferenceSession | null = null;
  private vocabularyCache = new Map<string, VocabularyPack>();
  private currentVocab: VocabularyPack | null = null;
  private config: Required<EncoderConfig>;
  private initialized = false;
  
  constructor(config: EncoderConfig = {}) {
    this.config = {
      modelUrl: config.modelUrl || '/models/universal_encoder.onnx',
      vocabUrl: config.vocabUrl || '/vocabs',
      wasmPaths: config.wasmPaths || '/wasm/',
      executionProviders: config.executionProviders || ['wasm', 'webgl'],
      graphOptimizationLevel: config.graphOptimizationLevel || 'all'
    };
    
    // Configure ONNX Runtime
    ort.env.wasm.wasmPaths = this.config.wasmPaths;
    
    // Only configure WebGL if it's in the execution providers
    if (this.config.executionProviders.includes('webgl')) {
      ort.env.webgl.powerPreference = 'high-performance';
    }
  }
  
  async initialize(): Promise<void> {
    if (this.initialized) return;
    
    try {
      console.log('🔄 Loading encoder model...');
      
      // Create session options
      const sessionOptions: ort.InferenceSession.SessionOptions = {
        executionProviders: this.config.executionProviders,
        graphOptimizationLevel: this.config.graphOptimizationLevel,
        enableCpuMemArena: false, // Disable for web
        enableMemPattern: false,  // Disable for web
      };
      
      // Load model
      this.session = await ort.InferenceSession.create(
        this.config.modelUrl,
        sessionOptions
      );
      
      this.initialized = true;
      
      console.log('✅ Encoder model loaded successfully');
      console.log('📊 Model inputs:', this.session.inputNames);
      console.log('📊 Model outputs:', this.session.outputNames);
      
      // Log execution providers being used
      console.log('🔧 Execution providers:', this.config.executionProviders);
      
    } catch (error) {
      console.error('❌ Failed to load encoder model:', error);
      throw new Error(`Failed to initialize encoder: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }
  
  async prepareTranslation(sourceLang: string, targetLang: string): Promise<void> {
    if (!this.initialized) {
      await this.initialize();
    }
    
    // Validate languages
    if (!LANGUAGE_TO_PACK[sourceLang]) {
      throw new Error(`Unsupported source language: ${sourceLang}`);
    }
    if (!LANGUAGE_TO_PACK[targetLang]) {
      throw new Error(`Unsupported target language: ${targetLang}`);
    }
    
    // Determine vocabulary pack (matching VocabularyManager logic)
    const sourcePack = LANGUAGE_TO_PACK[sourceLang];
    const targetPack = LANGUAGE_TO_PACK[targetLang];
    const packName = targetPack !== 'latin' ? targetPack : sourcePack;
    
    // Load vocabulary if not cached
    if (!this.vocabularyCache.has(packName)) {
      await this.loadVocabularyPack(packName);
    }
    
    this.currentVocab = this.vocabularyCache.get(packName)!;
    console.log(`📚 Using vocabulary pack: ${packName} for ${sourceLang} → ${targetLang}`);
  }
  
  private async loadVocabularyPack(packName: string): Promise<void> {
    try {
      const url = `${this.config.vocabUrl}/${packName}_v1.0.json`;
      console.log(`📥 Loading vocabulary pack from: ${url}`);
      
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      // Validate vocabulary structure
      if (!data.tokens || !data.subwords || !data.special_tokens) {
        throw new Error('Invalid vocabulary pack structure');
      }
      
      // Create vocabulary pack
      const pack: VocabularyPack = {
        name: data.name,
        version: data.version || '1.0',
        languages: data.languages || [],
        tokens: data.tokens,
        subwords: data.subwords,
        special_tokens: data.special_tokens,
        metadata: data.metadata || {
          total_tokens: Object.keys(data.tokens).length + 
                       Object.keys(data.subwords).length + 
                       Object.keys(data.special_tokens).length,
          size_mb: 0
        }
      };
      
      this.vocabularyCache.set(packName, pack);
      console.log(`✅ Loaded vocabulary pack: ${packName} (${pack.metadata.total_tokens} tokens)`);
      
    } catch (error) {
      console.error(`❌ Failed to load vocabulary pack ${packName}:`, error);
      throw new Error(`Failed to load vocabulary pack ${packName}: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }
  
  private tokenize(text: string, sourceLang: string): Int32Array {
    if (!this.currentVocab) {
      throw new Error('No vocabulary loaded');
    }
    
    const tokens: number[] = [];
    const vocab = this.currentVocab;
    
    // Add BOS token
    tokens.push(vocab.special_tokens['<s>'] || vocab.special_tokens['<bos>'] || 2);
    
    // Add language token if exists
    const langToken = `<${sourceLang}>`;
    if (vocab.special_tokens[langToken]) {
      tokens.push(vocab.special_tokens[langToken]);
    }
    
    // Normalize and tokenize text
    const normalizedText = text.toLowerCase().trim();
    const words = normalizedText.split(/\s+/).filter(w => w.length > 0);
    
    for (const word of words) {
      if (vocab.tokens[word]) {
        tokens.push(vocab.tokens[word]);
      } else {
        // Handle unknown words with subword tokenization
        const subwordTokens = this.tokenizeUnknown(word);
        tokens.push(...subwordTokens);
      }
    }
    
    // Add EOS token
    tokens.push(vocab.special_tokens['</s>'] || vocab.special_tokens['<eos>'] || 3);
    
    // Pad or truncate to 128 tokens (matching your system)
    const maxLength = 128;
    const padToken = vocab.special_tokens['<pad>'] || 0;
    
    if (tokens.length > maxLength) {
      // Keep EOS token at the end
      const truncated = tokens.slice(0, maxLength - 1);
      truncated.push(vocab.special_tokens['</s>'] || 3);
      return new Int32Array(truncated);
    } else {
      // Pad to max length
      while (tokens.length < maxLength) {
        tokens.push(padToken);
      }
      return new Int32Array(tokens);
    }
  }
  
  private tokenizeUnknown(word: string): number[] {
    if (!this.currentVocab) return [1]; // UNK token
    
    const tokens: number[] = [];
    const vocab = this.currentVocab;
    const unkToken = vocab.special_tokens['<unk>'] || 1;
    
    // Try to find subword matches (matching vocabulary_pack.cpp logic)
    let position = 0;
    
    while (position < word.length) {
      let found = false;
      
      // Try different subword lengths (from longest to shortest)
      for (let length = Math.min(word.length - position, 10); length > 0; length--) {
        const subword = '##' + word.slice(position, position + length);
        
        if (vocab.subwords[subword]) {
          tokens.push(vocab.subwords[subword]);
          position += length;
          found = true;
          break;
        }
      }
      
      if (!found) {
        // If no subword found, use UNK token and move one character
        tokens.push(unkToken);
        position++;
      }
    }
    
    return tokens.length > 0 ? tokens : [unkToken];
  }
  
  async encode(text: string, sourceLang: string, targetLang: string): Promise<Uint8Array> {
    if (!this.session) {
      throw new Error('Encoder not initialized. Call initialize() first.');
    }
    
    // Prepare translation (loads vocabulary)
    await this.prepareTranslation(sourceLang, targetLang);
    
    // Tokenize input
    const inputIds = this.tokenize(text, sourceLang);
    
    // Create attention mask
    const attentionMask = new Int32Array(128);
    const padToken = this.currentVocab?.special_tokens['<pad>'] || 0;
    for (let i = 0; i < 128; i++) {
      attentionMask[i] = inputIds[i] !== padToken ? 1 : 0;
    }
    
    // Create tensors
    const inputTensor = new ort.Tensor('int32', inputIds, [1, 128]);
    const maskTensor = new ort.Tensor('int32', attentionMask, [1, 128]);
    
    // Prepare inputs (matching your encoder's expected input names)
    const feeds: Record<string, ort.Tensor> = {};
    
    // Try different possible input names
    if (this.session.inputNames.includes('input_ids')) {
      feeds['input_ids'] = inputTensor;
    } else if (this.session.inputNames.includes('input')) {
      feeds['input'] = inputTensor;
    } else {
      feeds[this.session.inputNames[0]] = inputTensor;
    }
    
    if (this.session.inputNames.includes('attention_mask')) {
      feeds['attention_mask'] = maskTensor;
    } else if (this.session.inputNames.length > 1) {
      feeds[this.session.inputNames[1]] = maskTensor;
    }
    
    console.log('🔄 Running inference...');
    
    // Run inference
    const results = await this.session.run(feeds);
    
    // Get output tensor (try different possible output names)
    const outputNames = ['encoder_output', 'output', 'hidden_states', 'last_hidden_state'];
    let outputTensor: ort.Tensor | null = null;
    
    for (const name of outputNames) {
      if (results[name]) {
        outputTensor = results[name];
        break;
      }
    }
    
    if (!outputTensor && this.session.outputNames.length > 0) {
      outputTensor = results[this.session.outputNames[0]];
    }
    
    if (!outputTensor) {
      throw new Error('No encoder output found in results');
    }
    
    console.log('✅ Inference complete. Output shape:', outputTensor.dims);
    
    // Compress output (matching your C++ encoder format)
    return this.compressOutput(
      outputTensor.data as Float32Array,
      outputTensor.dims as number[]
    );
  }
  
  private compressOutput(data: Float32Array, shape: number[]): Uint8Array {
    // Shape should be [batch_size, sequence_length, hidden_dim]
    const sequenceLength = shape[1] || 128;
    const hiddenDim = shape[2] || 1024;
    
    console.log(`🔄 Compressing output: ${sequenceLength}x${hiddenDim}`);
    
    // Quantize to INT8 (matching your C++ implementation)
    let maxAbsVal = 0;
    for (let i = 0; i < data.length; i++) {
      maxAbsVal = Math.max(maxAbsVal, Math.abs(data[i]));
    }
    
    const scale = maxAbsVal > 0 ? 127.0 / maxAbsVal : 1.0;
    
    // Quantize
    const quantized = new Int8Array(data.length);
    for (let i = 0; i < data.length; i++) {
      const quantizedVal = Math.round(data[i] * scale);
      quantized[i] = Math.max(-128, Math.min(127, quantizedVal));
    }
    
    // Note: Your cloud decoder expects LZ4 compression, but LZ4 isn't readily available in browsers
    // We'll send uncompressed quantized data with a flag indicating no compression
    // The cloud decoder should be updated to handle this case
    
    // Create output with metadata header (16 bytes matching your format)
    const outputSize = 16 + quantized.length;
    const output = new Uint8Array(outputSize);
    const view = new DataView(output.buffer);
    
    // Write metadata
    view.setInt32(0, sequenceLength, true);     // sequence length
    view.setInt32(4, hiddenDim, true);          // hidden dimension  
    view.setFloat32(8, scale, true);             // scale factor
    view.setInt32(12, 0, true);                  // compression flag: 0 = uncompressed
    
    // Copy quantized data
    output.set(new Uint8Array(quantized.buffer), 16);
    
    console.log(`✅ Compressed: ${data.length * 4} bytes → ${output.length} bytes`);
    
    return output;
  }
  
  getSupportedLanguages(): string[] {
    return Object.keys(LANGUAGE_TO_PACK);
  }
  
  clearCache(): void {
    this.vocabularyCache.clear();
    this.currentVocab = null;
  }
}

export class TranslationClient {
  private encoder: TranslationEncoder;
  private decoderUrl: string;
  private cache = new Map<string, TranslationResult>();
  private maxCacheSize: number;
  private headers: Record<string, string>;
  
  constructor(options: {
    modelUrl?: string;
    decoderUrl?: string;
    maxCacheSize?: number;
    headers?: Record<string, string>;
  } = {}) {
    this.encoder = new TranslationEncoder({
      modelUrl: options.modelUrl
    });
    this.decoderUrl = options.decoderUrl || 'https://api.yourdomain.com/decode';
    this.maxCacheSize = options.maxCacheSize || 100;
    this.headers = options.headers || {};
  }
  
  async initialize(): Promise<void> {
    await this.encoder.initialize();
  }
  
  async translate(options: TranslationOptions): Promise<TranslationResult> {
    // Validate input
    if (!options.text?.trim()) {
      throw new Error('Text cannot be empty');
    }
    
    // Check cache
    const cacheKey = `${options.sourceLang}-${options.targetLang}:${options.text}`;
    const cached = this.cache.get(cacheKey);
    if (cached) {
      console.log('📦 Returning cached translation');
      return cached;
    }
    
    try {
      console.log(`🔄 Translating: "${options.text.substring(0, 50)}..." (${options.sourceLang} → ${options.targetLang})`);
      
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
          'X-Source-Language': options.sourceLang,
          ...this.headers
        },
        body: encoded,
      });
      
      if (!response.ok) {
        let errorMessage = `HTTP ${response.status}`;
        
        try {
          const errorData = await response.json();
          errorMessage = errorData.error || errorData.message || errorMessage;
        } catch {
          // If response is not JSON, try text
          try {
            errorMessage = await response.text();
          } catch {
            // Use default error message
          }
        }
        
        if (response.status === 429) {
          throw new Error('Rate limit exceeded. Please try again later.');
        } else if (response.status >= 500) {
          throw new Error('Translation service is temporarily unavailable.');
        } else {
          throw new Error(`Translation failed: ${errorMessage}`);
        }
      }
      
      const result = await response.json();
      
      const translationResult: TranslationResult = {
        translation: result.translation,
        targetLang: result.target_lang || options.targetLang,
        confidence: result.confidence
      };
      
      // Add to cache
      this.addToCache(cacheKey, translationResult);
      
      console.log('✅ Translation complete');
      
      return translationResult;
      
    } catch (error) {
      console.error('❌ Translation error:', error);
      
      if (error instanceof Error) {
        throw error;
      } else {
        throw new Error('An unexpected error occurred during translation');
      }
    }
  }
  
  async translateBatch(texts: string[], sourceLang: string, targetLang: string): Promise<TranslationResult[]> {
    const promises = texts.map(text => 
      this.translate({ text, sourceLang, targetLang })
    );
    
    return Promise.all(promises);
  }
  
  private addToCache(key: string, result: TranslationResult): void {
    // Implement LRU cache
    if (this.cache.size >= this.maxCacheSize) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
    this.cache.set(key, result);
  }
  
  clearCache(): void {
    this.cache.clear();
  }
  
  getSupportedLanguages(): string[] {
    return this.encoder.getSupportedLanguages();
  }
}

// Export for CommonJS compatibility
export default {
  TranslationEncoder,
  TranslationClient,
  LANGUAGE_TO_PACK
};