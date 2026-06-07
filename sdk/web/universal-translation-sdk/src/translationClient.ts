import { trace, context, propagation } from '@opentelemetry/api';
import { compress } from 'lz4js';
import { config } from './config';

const tracer = trace.getTracer('universal-translation-sdk');
const MODEL_VERSION = config.modelVersion;
const HF_REPO = config.hfRepo || 'your-org/universal-translation-system';

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

interface CoordinatorStatus {
  single_decoder: boolean;
  decoder_pool_size: number;
  healthy_decoders: number;
  decoders: Array<{ node_id: string; endpoint: string }>;
}

// Analytics class
class TranslationAnalytics {
  trackTranslation(sourceLang: string, targetLang: string, textLength: number, duration: number): void {
    console.log(`Analytics: Translation completed: ${sourceLang}->${targetLang}, ${textLength} chars, ${duration}ms`);
  }
  
  trackError(error: TranslationErrorCode, errorContext: Record<string, any>): void {
    console.error(`Analytics: Translation error: ${error}, context:`, errorContext);
  }
}

export interface TranslationClientOptions {
  decoderUrl?: string;
  coordinatorUrl?: string;
  localDecoderUrl?: string;
  preferLocal?: boolean;
  timeout?: number;
  retryCount?: number;
  enableAnalytics?: boolean;
  useWasm?: boolean;
}

export class TranslationClient {
  private decoderUrl: string;
  private coordinatorUrl: string | null = null;
  private localDecoderUrl: string | null = null;
  private preferLocal: boolean;
  private localDecoderAvailable: boolean = false;
  private useCoordinator: boolean = false;
  private timeout: number;
  private retryCount: number;
  private analytics: TranslationAnalytics;
  private enableAnalytics: boolean;
  private useWasm: boolean;
  private wasmEncoder: any = null;
  private wasmLoaded: boolean = false;
  private wasmLoading: Promise<void> | null = null;
  
  constructor(options: TranslationClientOptions = {}) {
    this.decoderUrl = options.decoderUrl || config.decoderApiUrl;
    this.coordinatorUrl = options.coordinatorUrl || config.coordinatorApiUrl || null;
    this.localDecoderUrl = options.localDecoderUrl || null;
    this.preferLocal = options.preferLocal !== false;
    this.timeout = options.timeout || 30000;
    this.retryCount = options.retryCount || 2;
    this.enableAnalytics = options.enableAnalytics !== false;
    this.useWasm = options.useWasm !== false;
    this.analytics = new TranslationAnalytics();
    
    if (this.useWasm) {
      this.loadWasmEncoder();
    }
    if (this.localDecoderUrl && this.preferLocal) {
      this.checkLocalDecoder();
    }
    if (this.coordinatorUrl) {
      this.resolveCoordinator();
    }
    this.checkEncoderUpdate();
  }
  
  get modelVersion() {
    return MODEL_VERSION;
  }

  private async resolveCoordinator(): Promise<void> {
    try {
      const resp = await fetch(`${this.coordinatorUrl}/api/status`, { signal: AbortSignal.timeout(5000) });
      if (!resp.ok) return;
      const status: CoordinatorStatus = await resp.json();
      if (status.single_decoder && status.decoders.length > 0) {
        this.decoderUrl = status.decoders[0].endpoint;
        this.useCoordinator = false;
      } else {
        this.useCoordinator = true;
      }
    } catch {
      console.warn('Coordinator unreachable, falling back to direct decoder');
    }
  }

  private async checkEncoderUpdate(): Promise<void> {
    try {
      const resp = await fetch(`https://huggingface.co/${HF_REPO}/raw/main/models/production/encoder.onnx`, {
        method: 'HEAD',
        signal: AbortSignal.timeout(5000)
      });
      if (!resp.ok) return;
      const remoteEtag = resp.headers.get('etag') || resp.headers.get('last-modified') || '';
      const cachedEtag = localStorage?.getItem('uts_encoder_etag');
      if (remoteEtag && remoteEtag !== cachedEtag) {
        localStorage?.setItem('uts_encoder_etag', remoteEtag);
        this.downloadEncoderUpdate();
      }
    } catch {
      // Offline or HF unavailable, bundled encoder is fine
    }
  }

  /**
   * Tries to discover a local decoder.
   * If localDecoderUrl is set, pings it directly.
   * Otherwise, scans common localhost ports.
   */
  private async checkLocalDecoder(): Promise<void> {
    if (this.localDecoderUrl) {
      try {
        const resp = await fetch(`${this.localDecoderUrl}/health`, { signal: AbortSignal.timeout(2000) });
        this.localDecoderAvailable = resp.ok;
        return;
      } catch {
        this.localDecoderAvailable = false;
        return;
      }
    }
    // Auto-scan common ports for a local decoder
    const ports = [8000, 8080, 9000];
    for (const port of ports) {
      try {
        const resp = await fetch(`http://localhost:${port}/health`, { signal: AbortSignal.timeout(1000) });
        if (resp.ok) {
          this.localDecoderUrl = `http://localhost:${port}/decode`;
          this.localDecoderAvailable = true;
          return;
        }
      } catch {
        continue;
      }
    }
    this.localDecoderAvailable = false;
  }

  private async getTargetUrl(): Promise<string> {
    if (this.localDecoderAvailable && this.localDecoderUrl && this.preferLocal) {
      return this.localDecoderUrl;
    }
    if (this.useCoordinator && this.coordinatorUrl) {
      return `${this.coordinatorUrl}/api/decode`;
    }
    return this.decoderUrl;
  }

  private async downloadEncoderUpdate(): Promise<void> {
    try {
      const resp = await fetch(`https://huggingface.co/${HF_REPO}/resolve/main/models/production/encoder.onnx`, {
        signal: AbortSignal.timeout(30000)
      });
      if (!resp.ok) return;
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      localStorage?.setItem('uts_encoder_url', url);
    } catch {
      console.warn('Failed to download encoder update, using bundled version');
    }
  }

  private async loadWasmEncoder(): Promise<void> {
    if (this.wasmLoaded || this.wasmLoading) {
      return this.wasmLoading;
    }
    
    this.wasmLoading = new Promise<void>(async (resolve, reject) => {
      try {
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
    
    const baseUrl = this.localDecoderAvailable && this.preferLocal && this.localDecoderUrl
      ? this.localDecoderUrl
      : this.useCoordinator && this.coordinatorUrl
        ? this.coordinatorUrl
        : this.decoderUrl;
    
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'X-Model-Version': MODEL_VERSION,
      'X-Source-Language': sourceLang,
      'X-Target-Language': targetLang
    };
    
    // Inject OpenTelemetry context
    propagation.inject(context.active(), headers);
    
    try {
      const response = await this.fetchWithRetry(`${baseUrl}/translate`, {
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
   * Sends encoded data to decoder (or coordinator when multi-decoder)
   */
  private async sendToDecoder(
    compressedData: Uint8Array,
    sourceLang: string,
    targetLang: string,
    startTime: number,
    span: any
  ): Promise<TranslationResult> {
    span.setAttribute('encoding_method', 'local');
    
    const targetUrl = this.useCoordinator && this.coordinatorUrl && !(this.localDecoderAvailable && this.preferLocal)
      ? `${this.coordinatorUrl}/api/decode`
      : this.localDecoderAvailable && this.preferLocal && this.localDecoderUrl
        ? this.localDecoderUrl
        : this.decoderUrl;
    
    const headers: Record<string, string> = {
      'Content-Type': 'application/octet-stream',
      'X-Model-Version': MODEL_VERSION,
      'X-Source-Language': sourceLang,
      'X-Target-Language': targetLang
    };
    
    // Inject OpenTelemetry context
    propagation.inject(context.active(), headers);
    
    try {
      const response = await this.fetchWithRetry(targetUrl, {
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