/**
 * WebAssembly encoder wrapper for the Universal Translation SDK
 */
import { config } from './config';

export interface WasmEncoderOptions {
  /** Path to the WebAssembly module */
  wasmPath?: string;
  /** Whether to use the WebAssembly module */
  useWasm?: boolean;
  /** API endpoint for cloud fallback */
  cloudApiEndpoint?: string;
  /** Whether to enable automatic fallback to cloud API */
  enableFallback?: boolean;
}

export class WasmEncoderWrapper {
  private wasmModule: any = null;
  private encoder: any = null;
  private loading: Promise<void> | null = null;
  private options: WasmEncoderOptions;
  
  constructor(options: WasmEncoderOptions = {}) {
    this.options = {
      wasmPath: options.wasmPath || config.wasmEncoderPath,
      useWasm: options.useWasm ?? config.useWasmEncoder,
      cloudApiEndpoint: options.cloudApiEndpoint || config.encoderApiUrl,
      enableFallback: options.enableFallback ?? config.enableFallback
    };
  }
  
  /**
   * Load the WebAssembly module
   */
  async load(): Promise<void> {
    if (!this.options.useWasm) {
      throw new Error('WebAssembly is disabled');
    }
    
    if (this.encoder) {
      return Promise.resolve();
    }
    
    if (this.loading) {
      return this.loading;
    }
    
    this.loading = new Promise<void>(async (resolve, reject) => {
      try {
        // Import the WASM module
        const wasmPath = this.options.wasmPath || '../public/wasm/encoder.js';
        
        // Dynamic import
        const createWasmEncoder = await import(/* webpackIgnore: true */ wasmPath);
        this.wasmModule = await createWasmEncoder.default();
        
        // Create encoder instance
        this.encoder = new this.wasmModule.WasmEncoder();
        
        // Initialize
        const success = this.encoder.initialize();
        if (!success) {
          throw new Error('Failed to initialize WASM encoder');
        }
        
        console.log('WASM encoder initialized successfully');
        resolve();
      } catch (error) {
        console.error('Failed to load WASM encoder:', error);
        this.encoder = null;
        this.wasmModule = null;
        reject(error);
      } finally {
        this.loading = null;
      }
    });
    
    return this.loading;
  }
  
  /**
   * Check if the WebAssembly module is loaded
   */
  isLoaded(): boolean {
    return !!this.encoder;
  }
  
  /**
   * Load vocabulary for a language
   * @param language Language code
   */
  async loadVocabulary(language: string): Promise<boolean> {
    if (!this.encoder) {
      await this.load();
    }
    
    return this.encoder.loadVocabulary(language);
  }
  
  /**
   * Check if vocabulary is loaded for a language
   * @param language Language code
   */
  hasVocabulary(language: string): boolean {
    if (!this.encoder) {
      return false;
    }
    
    return this.encoder.hasVocabulary(language);
  }
  
  /**
   * Encode text to embeddings
   * @param text Text to encode
   * @param sourceLang Source language
   * @param targetLang Target language
   */
  async encode(text: string, sourceLang: string, targetLang: string): Promise<Float32Array> {
    if (!this.encoder) {
      await this.load();
    }
    
    // Check if vocabulary is loaded
    if (!this.hasVocabulary(sourceLang)) {
      await this.loadVocabulary(sourceLang);
    }
    
    // Encode the text
    const result = this.encoder.encode(text, sourceLang, targetLang);
    
    // Convert to Float32Array
    return new Float32Array(result);
  }
  
  /**
   * Compress embeddings for transmission
   * @param embedding Embeddings to compress
   */
  async compressEmbedding(embedding: Float32Array): Promise<Uint8Array> {
    if (!this.encoder) {
      await this.load();
    }
    
    // Compress the embedding
    const result = this.encoder.compressEmbedding(embedding);
    
    // Convert to Uint8Array
    return new Uint8Array(result);
  }
  
  /**
   * Get supported languages
   */
  async getSupportedLanguages(): Promise<string[]> {
    if (!this.encoder) {
      await this.load();
    }
    
    // Get supported languages
    const result = this.encoder.getSupportedLanguages();
    
    // Convert to string array
    const languages: string[] = [];
    for (let i = 0; i < result.length; i++) {
      languages.push(result[i]);
    }
    
    return languages;
  }
  
  /**
   * Encode text to embeddings with fallback to cloud API if WASM fails
   * @param text Text to encode
   * @param sourceLang Source language
   * @param targetLang Target language
   */
  async encodeWithFallback(text: string, sourceLang: string, targetLang: string): Promise<Float32Array> {
    try {
      return await this.encode(text, sourceLang, targetLang);
    } catch (error) {
      console.warn('WASM encoding failed, falling back to cloud API:', error);
      
      if (!this.options.enableFallback) {
        throw new Error('WASM encoding failed and fallback is disabled');
      }
      
      // Implement fallback to cloud API
      return await this.fallbackToCloudEncoder(text, sourceLang, targetLang);
    }
  }

  /**
   * Fallback to cloud API for encoding
   * @param text Text to encode
   * @param sourceLang Source language
   * @param targetLang Target language
   * @private
   */
  private async fallbackToCloudEncoder(text: string, sourceLang: string, targetLang: string): Promise<Float32Array> {
    const endpoint = this.options.cloudApiEndpoint || config.encoderApiUrl;
    
    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, sourceLang, targetLang })
      });
      
      if (!response.ok) {
        throw new Error(`Cloud encoding failed: ${response.statusText}`);
      }
      
      const data = await response.json();
      return new Float32Array(data.embedding);
    } catch (error) {
      console.error('Cloud fallback failed:', error);
      throw new Error(`Both WASM and cloud encoding failed: ${error.message}`);
    }
  }
  
  /**
   * Destroy the encoder and free resources
   */
  destroy(): void {
    if (this.encoder) {
      this.encoder.destroy();
      this.encoder = null;
    }
    
    this.wasmModule = null;
  }
}