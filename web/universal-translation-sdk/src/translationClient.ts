import { trace, context, propagation } from '@opentelemetry/api';
import { compress } from 'lz4js';

const tracer = trace.getTracer('universal-translation-sdk');
const MODEL_VERSION = process.env.MODEL_VERSION || '1.0.0';
const DEFAULT_DECODER_URL = 'https://api.yourdomain.com/decode';

// Error codes enum - matching other SDKs
export enum TranslationErrorCode {
  NETWORK_ERROR = "NETWORK_ERROR",
  MODEL_NOT_FOUND = "MODEL_NOT_FOUND",
  VOCABULARY_NOT_LOADED = "VOCABULARY_NOT_LOADED",
  ENCODING_FAILED = "ENCODING_FAILED",
  DECODING_FAILED = "DECODING_FAILED",
  INVALID_LANGUAGE = "INVALID_LANGUAGE",
  RATE_LIMITED = "RATE_LIMITED"
}

// Result types
export type TranslationSuccess = {
  translation: string;
  sourceLang: string;
  targetLang: string;
  confidence?: number;
  duration: number;
};

export type TranslationError = {
  message: string;
  code: TranslationErrorCode;
  details?: any;
};

export type TranslationResult = 
  | { success: true; data: TranslationSuccess }
  | { success: false; error: TranslationError };

// Analytics class
class TranslationAnalytics {
  trackTranslation(sourceLang: string, targetLang: string, textLength: number, duration: number): void {
    console.log(`Analytics: Translation completed: ${sourceLang}->${targetLang}, ${textLength} chars, ${duration}ms`);
    // Implement your analytics tracking here
    // Example: Google Analytics, Amplitude, etc.
  }
  
  trackError(error: TranslationErrorCode, context: Record<string, any>): void {
    console.error(`Analytics: Translation error: ${error}, context:`, context);
    // Implement your error tracking here
  }
}

export interface TranslationClientOptions {
  decoderUrl?: string;
  timeout?: number;
  retryCount?: number;
  enableAnalytics?: boolean;
  useWasm?: boolean; // Whether to use WebAssembly for encoding (if available)
}

export class TranslationClient {
  private decoderUrl: string;
  private timeout: number;
  private retryCount: number;
  private analytics: TranslationAnalytics;
  private enableAnalytics: boolean;
  private useWasm: boolean;
  private wasmEncoder: any = null;
  private wasmLoaded: boolean = false;
  private wasmLoading: Promise<void> | null = null;
  
  constructor(options: TranslationClientOptions = {}) {
    this.decoderUrl = options.decoderUrl || DEFAULT_DECODER_URL;
    this.timeout = options.timeout || 30000; // 30 seconds default
    this.retryCount = options.retryCount || 2;
    this.enableAnalytics = options.enableAnalytics !== false;
    this.useWasm = options.useWasm !== false;
    this.analytics = new TranslationAnalytics();
    
    // Try to load WASM encoder if enabled
    if (this.useWasm) {
      this.loadWasmEncoder();
    }
  }
  
  get modelVersion() {
    return MODEL_VERSION;
  }

  /**
   * Loads the WebAssembly encoder module
   */
  private async loadWasmEncoder(): Promise<void> {
    if (this.wasmLoaded || this.wasmLoading) {
      return this.wasmLoading;
    }
    
    this.wasmLoading = new Promise<void>(async (resolve, reject) => {
      try {
        // In a real implementation, you would load your WASM module here
        // Example:
        // const module = await import('@universal-translation/wasm-encoder');
        // this.wasmEncoder = module.createEncoder();
        
        // For now, we'll just simulate it
        await new Promise(resolve => setTimeout(resolve, 100));
        this.wasmLoaded = true;
        resolve();
      } catch (error) {
        console.warn('Failed to load WASM encoder:', error);
        this.wasmLoaded = false;
        this.wasmEncoder = null;
        reject(error);
      } finally {
        this.wasmLoading = null;
      }
    });
    
    return this.wasmLoading;
  }

  /**
   * Translates text from source language to target language
   */
  async translate({ 
    text, 
    sourceLang, 
    targetLang 
  }: { 
    text: string; 
    sourceLang: string; 
    targetLang: string 
  }): Promise<TranslationResult> {
    const span = tracer.startSpan('TranslationClient.translate');
    span.setAttribute('model_version', MODEL_VERSION);
    span.setAttribute('source_lang', sourceLang);
    span.setAttribute('target_lang', targetLang);
    
    const startTime = Date.now();
    
    try {
      // Validate input
      if (!text || text.trim() === '') {
        throw new Error('Text cannot be empty');
      }
      
      if (!sourceLang || !targetLang) {
        throw new Error('Source and target languages must be specified');
      }
      
      // Try to encode locally if WASM is available
      let encodedData: Uint8Array | null = null;
      
      if (this.useWasm && this.wasmEncoder) {
        try {
          // Try to encode using WASM
          encodedData = await this.encodeWithWasm(text, sourceLang, targetLang);
        } catch (encodeError) {
          console.warn('WASM encoding failed, falling back to API', encodeError);
        }
      }
      
      // If local encoding failed or isn't available, use the API directly
      if (!encodedData) {
        // Send to API for encoding and translation in one step
        return await this.translateViaAPI(text, sourceLang, targetLang, startTime, span);
      }
      
      // Compress the encoded data
      const compressedData = compress(encodedData);
      
      // Send to decoder
      return await this.sendToDecoder(compressedData, sourceLang, targetLang, startTime, span);
    } catch (error) {
      const duration = Date.now() - startTime;
      const errorCode = this.getErrorCodeFromException(error);
      
      if (this.enableAnalytics) {
        this.analytics.trackError(errorCode, {
          sourceLang,
          targetLang,
          errorMessage: error.message || 'Unknown error',
          duration
        });
      }
      
      span.setAttribute('error', true);
      span.setAttribute('error.message', error.message || 'Unknown error');
      span.end();
      
      return {
        success: false,
        error: {
          message: error.message || 'Translation failed',
          code: errorCode,
          details: error
        }
      };
    }
  }
  
  /**
   * Encodes text using WebAssembly
   */
  private async encodeWithWasm(text: string, sourceLang: string, targetLang: string): Promise<Uint8Array | null> {
    if (!this.wasmLoaded && this.useWasm) {
      await this.loadWasmEncoder();
    }
    
    if (!this.wasmEncoder) {
      return null;
    }
    
    // In a real implementation, you would call your WASM encoder here
    // Example:
    // return this.wasmEncoder.encode(text, sourceLang, targetLang);
    
    // For now, we'll just return null to fall back to API
    return null;
  }
  
  /**
   * Translates directly via API when local encoding isn't available
   */
  private async translateViaAPI(
    text: string,
    sourceLang: string,
    targetLang: string,
    startTime: number,
    span: any
  ): Promise<TranslationResult> {
    span.setAttribute('encoding_method', 'api');
    
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'X-Model-Version': MODEL_VERSION,
      'X-Source-Language': sourceLang,
      'X-Target-Language': targetLang
    };
    
    // Inject OpenTelemetry context
    propagation.inject(context.active(), headers);
    
    try {
      const response = await this.fetchWithRetry(`${this.decoderUrl}/translate`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          text,
          sourceLang,
          targetLang
        })
      });
      
      if (!response.ok) {
        if (response.status === 429) {
          throw new Error('Rate limit exceeded. Please try again later.');
        } else if (response.status >= 500) {
          throw new Error('Server error. Please try again later.');
        } else {
          throw new Error(`Translation failed with status: ${response.status}`);
        }
      }
      
      const result = await response.json();
      const duration = Date.now() - startTime;
      
      if (this.enableAnalytics) {
        this.analytics.trackTranslation(sourceLang, targetLang, text.length, duration);
      }
      
      span.end();
      
      return {
        success: true,
        data: {
          translation: result.translation,
          sourceLang,
          targetLang,
          confidence: result.confidence,
          duration
        }
      };
    } catch (error) {
      throw error; // Will be caught by the outer try/catch
    }
  }
  
  /**
   * Sends encoded data to decoder
   */
  private async sendToDecoder(
    compressedData: Uint8Array,
    sourceLang: string,
    targetLang: string,
    startTime: number,
    span: any
  ): Promise<TranslationResult> {
    span.setAttribute('encoding_method', 'local');
    
    const headers: Record<string, string> = {
      'Content-Type': 'application/octet-stream',
      'X-Model-Version': MODEL_VERSION,
      'X-Source-Language': sourceLang,
      'X-Target-Language': targetLang
    };
    
    // Inject OpenTelemetry context
    propagation.inject(context.active(), headers);
    
    try {
      const response = await this.fetchWithRetry(this.decoderUrl, {
        method: 'POST',
        headers,
        body: compressedData
      });
      
      if (!response.ok) {
        if (response.status === 429) {
          throw new Error('Rate limit exceeded. Please try again later.');
        } else if (response.status >= 500) {
          throw new Error('Server error. Please try again later.');
        } else {
          throw new Error(`Translation failed with status: ${response.status}`);
        }
      }
      
      const result = await response.json();
      const duration = Date.now() - startTime;
      
      if (this.enableAnalytics) {
        this.analytics.trackTranslation(sourceLang, targetLang, result.originalLength || 0, duration);
      }
      
      span.end();
      
      return {
        success: true,
        data: {
          translation: result.translation,
          sourceLang,
          targetLang,
          confidence: result.confidence,
          duration
        }
      };
    } catch (error) {
      throw error; // Will be caught by the outer try/catch
    }
  }
  
  /**
   * Fetch with retry logic
   */
  private async fetchWithRetry(url: string, options: RequestInit, retryCount = this.retryCount): Promise<Response> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);
      
      const response = await fetch(url, {
        ...options,
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      return response;
    } catch (error) {
      if (error.name === 'AbortError') {
        throw new Error('Request timed out');
      }
      
      if (retryCount <= 0) {
        throw error;
      }
      
      // Exponential backoff
      const delay = Math.pow(2, this.retryCount - retryCount) * 1000;
      await new Promise(resolve => setTimeout(resolve, delay));
      
      return this.fetchWithRetry(url, options, retryCount - 1);
    }
  }
  
  /**
   * Maps exceptions to error codes
   */
  private getErrorCodeFromException(error: any): TranslationErrorCode {
    if (!error) {
      return TranslationErrorCode.ENCODING_FAILED;
    }
    
    const message = error.message || '';
    
    if (message.includes('timed out') || message.includes('network') || error.name === 'NetworkError') {
      return TranslationErrorCode.NETWORK_ERROR;
    }
    
    if (message.includes('vocabulary') || message.includes('vocab')) {
      return TranslationErrorCode.VOCABULARY_NOT_LOADED;
    }
    
    if (message.includes('model')) {
      return TranslationErrorCode.MODEL_NOT_FOUND;
    }
    
    if (message.includes('Rate limit')) {
      return TranslationErrorCode.RATE_LIMITED;
    }
    
    if (message.includes('language')) {
      return TranslationErrorCode.INVALID_LANGUAGE;
    }
    
    return TranslationErrorCode.ENCODING_FAILED;
  }
}