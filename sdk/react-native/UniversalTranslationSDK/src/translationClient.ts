import { trace, context, propagation } from '@opentelemetry/api';
import { NativeModules, Platform } from 'react-native';
import LZ4 from 'react-native-lz4';
import { config } from './config';

const tracer = trace.getTracer('universal-translation-sdk');
const MODEL_VERSION = config.modelVersion;
const HF_REPO = config.hfRepo;

interface CoordinatorStatus {
  single_decoder: boolean;
  decoder_pool_size: number;
  healthy_decoders: number;
  decoders: { node_id: string; endpoint: string }[];
}

// Error codes enum - matching Android implementation
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
    // Example: Firebase Analytics, Amplitude, etc.
  }
  
  trackError(error: TranslationErrorCode, context: Record<string, any>): void {
    console.error(`Analytics: Translation error: ${error}, context:`, context);
    // Implement your error tracking here
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
}

export class TranslationClient {
  private decoderUrl: string;
  private coordinatorUrl: string | null;
  private localDecoderUrl: string | null;
  private preferLocal: boolean;
  private localDecoderAvailable: boolean = false;
  private useCoordinator: boolean = false;
  private timeout: number;
  private retryCount: number;
  private analytics: TranslationAnalytics;
  private enableAnalytics: boolean;
  
  constructor(options: TranslationClientOptions = {}) {
    this.decoderUrl = options.decoderUrl || config.decoderApiUrl;
    this.coordinatorUrl = options.coordinatorUrl || config.coordinatorApiUrl || null;
    this.localDecoderUrl = options.localDecoderUrl || null;
    this.preferLocal = options.preferLocal !== false;
    this.timeout = options.timeout || 30000;
    this.retryCount = options.retryCount || 2;
    this.enableAnalytics = options.enableAnalytics !== false;
    this.analytics = new TranslationAnalytics();
    
    if (this.localDecoderUrl && this.preferLocal) {
      this.checkLocalDecoder();
    }
    if (this.coordinatorUrl) {
      this.resolveCoordinator();
    }
    this.checkEncoderUpdate();
  }

  private async checkLocalDecoder(): Promise<void> {
    try {
      const resp = await fetch(`${this.localDecoderUrl}/health`, { signal: AbortSignal.timeout(2000) });
      this.localDecoderAvailable = resp.ok;
    } catch {
      this.localDecoderAvailable = false;
    }
  }

  private getBaseUrl(): string {
    if (this.localDecoderAvailable && this.preferLocal && this.localDecoderUrl) {
      return this.localDecoderUrl;
    }
    if (this.useCoordinator && this.coordinatorUrl) {
      return this.coordinatorUrl;
    }
    return this.decoderUrl;
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
      const cachedEtag = await this.getCachedEtag();
      if (remoteEtag && remoteEtag !== cachedEtag) {
        await this.cacheEtag(remoteEtag);
        this.downloadEncoderUpdate();
      }
    } catch {
      // Offline or HF unavailable, bundled encoder is fine
    }
  }

  private async getCachedEtag(): Promise<string | null> {
    try {
      const val = await NativeModules?.UniversalTranslationEncoder?.getCachedEtag?.();
      return val || null;
    } catch {
      return null;
    }
  }

  private async cacheEtag(etag: string): Promise<void> {
    try {
      await NativeModules?.UniversalTranslationEncoder?.cacheEtag?.(etag);
    } catch {
      // Non-critical, bundled encoder works fine
    }
  }

  private async downloadEncoderUpdate(): Promise<void> {
    try {
      await NativeModules?.UniversalTranslationEncoder?.downloadEncoderUpdate?.();
    } catch {
      console.warn('Failed to download encoder update, using bundled version');
    }
  }
  
  get modelVersion() {
    return MODEL_VERSION;
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
      
      // Try to encode locally if native module is available
      let encodedData: Uint8Array | null = null;
      
      try {
        if (NativeModules.UniversalTranslationEncoder) {
          // Use native encoder if available
          const nativeResult = await NativeModules.UniversalTranslationEncoder.encode(
            text,
            sourceLang,
            targetLang
          );
          
          if (nativeResult && nativeResult.data) {
            encodedData = new Uint8Array(nativeResult.data);
          }
        }
      } catch (encodeError) {
        console.warn('Native encoding failed, falling back to API', encodeError);
        // Continue with API-based encoding
      }
      
      // If local encoding failed or isn't available, use the API directly
      if (!encodedData) {
        // Send to API for encoding and translation in one step
        return await this.translateViaAPI(text, sourceLang, targetLang, startTime, span);
      }
      
      // Compress the encoded data
      const compressedData = await LZ4.compress(encodedData);
      
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
    
    const baseUrl = this.getBaseUrl();
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
   * Sends encoded data to decoder (or coordinator when multi-decoder, or local decoder when available)
   */
  private async sendToDecoder(
    compressedData: Uint8Array,
    sourceLang: string,
    targetLang: string,
    startTime: number,
    span: any
  ): Promise<TranslationResult> {
    span.setAttribute('encoding_method', 'local');
    
    const targetUrl = this.localDecoderAvailable && this.preferLocal && this.localDecoderUrl
      ? this.localDecoderUrl
      : this.useCoordinator && this.coordinatorUrl
        ? `${this.coordinatorUrl}/api/decode`
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
  
  /**
   * Gets performance metrics
   */
  getPerformanceMetrics(): any {
    // If native module is available, get metrics from it
    if (NativeModules.UniversalTranslationEncoder && 
        typeof NativeModules.UniversalTranslationEncoder.getPerformanceMetrics === 'function') {
      return NativeModules.UniversalTranslationEncoder.getPerformanceMetrics();
    }
    
    return {};
  }
  
  /**
   * Cleans up resources
   */
  destroy(): void {
    // If native module is available, call destroy
    if (NativeModules.UniversalTranslationEncoder && 
        typeof NativeModules.UniversalTranslationEncoder.destroy === 'function') {
      NativeModules.UniversalTranslationEncoder.destroy();
    }
  }
}