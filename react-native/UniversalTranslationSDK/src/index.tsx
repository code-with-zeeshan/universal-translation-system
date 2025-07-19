// react-native/UniversalTranslationSDK/src/index.tsx

import {
  NativeModules,
  Platform,
  NativeEventEmitter,
  EmitterSubscription,
} from 'react-native';
import React, { useState, useCallback, useEffect, useRef } from 'react';

// Types
export interface TranslationOptions {
  text: string;
  sourceLang: string;
  targetLang: string;
  options?: {
    formality?: 'formal' | 'informal' | 'auto';
    domain?: 'general' | 'medical' | 'legal' | 'technical' | 'business';
    preserveFormatting?: boolean;
  };
}

export interface TranslationResult {
  translation: string;
  confidence?: number;
  alternativeTranslations?: string[];
  detectedSourceLang?: string;
}

export interface VocabularyInfo {
  name: string;
  languages: string[];
  sizeMB: number;
  isDownloaded: boolean;
  version: string;
  downloadProgress?: number;
}

export interface LanguageInfo {
  code: string;
  name: string;
  nativeName: string;
  isRTL: boolean;
}

export interface TranslationError extends Error {
  code: string;
  userMessage: string;
}

// Native Module Interface
interface IUniversalTranslationModule {
  initialize(): Promise<void>;
  prepareTranslation(sourceLang: string, targetLang: string): Promise<void>;
  encode(text: string, sourceLang: string, targetLang: string): Promise<string>;
  getAvailableVocabularies(): Promise<VocabularyInfo[]>;
  downloadVocabulary(name: string): Promise<void>;
  deleteVocabulary(name: string): Promise<void>;
  getSupportedLanguages(): Promise<LanguageInfo[]>;
  getMemoryUsage(): Promise<number>;
  clearCache(): Promise<void>;
}

const LINKING_ERROR =
  `The package '@universal-translation/react-native-sdk' doesn't seem to be linked. Make sure: \n\n` +
  Platform.select({ ios: "- You have run 'pod install'\n", default: '' }) +
  '- You rebuilt the app after installing the package\n' +
  '- You are not using Expo managed workflow\n';

const UniversalTranslationModule = NativeModules.UniversalTranslationModule
  ? NativeModules.UniversalTranslationModule
  : new Proxy(
      {},
      {
        get() {
          throw new Error(LINKING_ERROR);
        },
      }
    ) as IUniversalTranslationModule;

// Event Emitter for download progress
const eventEmitter = new NativeEventEmitter(NativeModules.UniversalTranslationModule);

// Translation Encoder Class
class TranslationEncoder {
  private initialized: boolean = false;
  private initPromise: Promise<void> | null = null;

  async initialize(): Promise<void> {
    if (this.initialized) return;
    
    if (this.initPromise) {
      return this.initPromise;
    }

    this.initPromise = UniversalTranslationModule.initialize()
      .then(() => {
        this.initialized = true;
      })
      .catch((error) => {
        this.initPromise = null;
        throw this.createError(error);
      });

    return this.initPromise;
  }

  async prepareTranslation(sourceLang: string, targetLang: string): Promise<void> {
    await this.initialize();
    
    try {
      return await UniversalTranslationModule.prepareTranslation(sourceLang, targetLang);
    } catch (error) {
      throw this.createError(error);
    }
  }

  async encode(text: string, sourceLang: string, targetLang: string): Promise<Uint8Array> {
    await this.initialize();
    
    try {
      const base64 = await UniversalTranslationModule.encode(text, sourceLang, targetLang);
      
      // Convert base64 to Uint8Array
      const binaryString = atob(base64);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      return bytes;
    } catch (error) {
      throw this.createError(error);
    }
  }

  async getAvailableVocabularies(): Promise<VocabularyInfo[]> {
    await this.initialize();
    
    try {
      return await UniversalTranslationModule.getAvailableVocabularies();
    } catch (error) {
      throw this.createError(error);
    }
  }

  async downloadVocabulary(
    name: string,
    onProgress?: (progress: number) => void
  ): Promise<void> {
    await this.initialize();
    
    let progressListener: EmitterSubscription | null = null;
    
    try {
      if (onProgress) {
        progressListener = eventEmitter.addListener(
          'vocabularyDownloadProgress',
          (event) => {
            if (event.name === name) {
              onProgress(event.progress);
            }
          }
        );
      }
      
      await UniversalTranslationModule.downloadVocabulary(name);
    } catch (error) {
      throw this.createError(error);
    } finally {
      progressListener?.remove();
    }
  }

  async deleteVocabulary(name: string): Promise<void> {
    await this.initialize();
    
    try {
      return await UniversalTranslationModule.deleteVocabulary(name);
    } catch (error) {
      throw this.createError(error);
    }
  }

  async getSupportedLanguages(): Promise<LanguageInfo[]> {
    await this.initialize();
    
    try {
      return await UniversalTranslationModule.getSupportedLanguages();
    } catch (error) {
      throw this.createError(error);
    }
  }

  async getMemoryUsage(): Promise<number> {
    try {
      return await UniversalTranslationModule.getMemoryUsage();
    } catch (error) {
      throw this.createError(error);
    }
  }

  async clearCache(): Promise<void> {
    try {
      return await UniversalTranslationModule.clearCache();
    } catch (error) {
      throw this.createError(error);
    }
  }

  private createError(error: any): TranslationError {
    const translationError = new Error(error.message || 'Unknown error') as TranslationError;
    translationError.code = error.code || 'UNKNOWN_ERROR';
    translationError.userMessage = error.userMessage || error.message || 'An error occurred';
    return translationError;
  }
}

// Translation Client Class
export class TranslationClient {
  private encoder: TranslationEncoder;
  private decoderUrl: string;
  private headers: Record<string, string>;
  private timeout: number;
  private cache: Map<string, TranslationResult>;
  private maxCacheSize: number;

  constructor(options?: {
    decoderUrl?: string;
    headers?: Record<string, string>;
    timeout?: number;
    maxCacheSize?: number;
  }) {
    this.encoder = new TranslationEncoder();
    this.decoderUrl = options?.decoderUrl || 'https://api.yourdomain.com/decode';
    this.headers = options?.headers || {};
    this.timeout = options?.timeout || 30000;
    this.maxCacheSize = options?.maxCacheSize || 100;
    this.cache = new Map();
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
      return cached;
    }

    try {
      // Encode locally
      const encoded = await this.encoder.encode(
        options.text,
        options.sourceLang,
        options.targetLang
      );

      // Create abort controller for timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);

      try {
        // Send to decoder
        const response = await fetch(this.decoderUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/octet-stream',
            'X-Target-Language': options.targetLang,
            'X-Source-Language': options.sourceLang,
            ...this.headers,
          },
          body: encoded,
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`Translation failed (${response.status}): ${errorText}`);
        }

        const result: TranslationResult = await response.json();
        
        // Add to cache
        this.addToCache(cacheKey, result);
        
        return result;
      } finally {
        clearTimeout(timeoutId);
      }
    } catch (error: any) {
      if (error.name === 'AbortError') {
        throw new Error('Translation request timed out');
      }
      throw new Error(`Translation error: ${error.message}`);
    }
  }

  async translateBatch(
    texts: string[],
    sourceLang: string,
    targetLang: string
  ): Promise<TranslationResult[]> {
    const promises = texts.map((text) =>
      this.translate({ text, sourceLang, targetLang })
    );
    
    return Promise.all(promises);
  }

  async getVocabularyInfo(): Promise<VocabularyInfo[]> {
    return this.encoder.getAvailableVocabularies();
  }

  async downloadVocabulary(
    name: string,
    onProgress?: (progress: number) => void
  ): Promise<void> {
    return this.encoder.downloadVocabulary(name, onProgress);
  }

  async deleteVocabulary(name: string): Promise<void> {
    return this.encoder.deleteVocabulary(name);
  }

  async getSupportedLanguages(): Promise<LanguageInfo[]> {
    return this.encoder.getSupportedLanguages();
  }

  async getMemoryUsage(): Promise<number> {
    return this.encoder.getMemoryUsage();
  }

  clearCache(): void {
    this.cache.clear();
  }

  async clearNativeCache(): Promise<void> {
    return this.encoder.clearCache();
  }

  private addToCache(key: string, result: TranslationResult): void {
    // Implement LRU cache
    if (this.cache.size >= this.maxCacheSize) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
    this.cache.set(key, result);
  }
}

// React Hook for easy usage
export function useTranslation(options?: {
  decoderUrl?: string;
  headers?: Record<string, string>;
  timeout?: number;
}) {
  const clientRef = useRef<TranslationClient>();
  const [isTranslating, setIsTranslating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<number>(0);

  // Initialize client
  useEffect(() => {
    clientRef.current = new TranslationClient(options);
  }, [options?.decoderUrl, options?.timeout]);

  const translate = useCallback(async (translationOptions: TranslationOptions) => {
    if (!clientRef.current) {
      throw new Error('Translation client not initialized');
    }

    setIsTranslating(true);
    setError(null);

    try {
      const result = await clientRef.current.translate(translationOptions);
      return result;
    } catch (err: any) {
      const errorMessage = err.userMessage || err.message || 'Translation failed';
      setError(errorMessage);
      throw err;
    } finally {
      setIsTranslating(false);
    }
  }, []);

  const downloadVocabulary = useCallback(async (name: string) => {
    if (!clientRef.current) {
      throw new Error('Translation client not initialized');
    }

    setProgress(0);
    
    try {
      await clientRef.current.downloadVocabulary(name, (p) => {
        setProgress(p);
      });
      setProgress(100);
    } catch (err: any) {
      const errorMessage = err.userMessage || err.message || 'Download failed';
      setError(errorMessage);
      throw err;
    }
  }, []);

  const getVocabularyInfo = useCallback(async () => {
    if (!clientRef.current) {
      throw new Error('Translation client not initialized');
    }

    return clientRef.current.getVocabularyInfo();
  }, []);

  const getSupportedLanguages = useCallback(async () => {
    if (!clientRef.current) {
      throw new Error('Translation client not initialized');
    }

    return clientRef.current.getSupportedLanguages();
  }, []);

  const clearCache = useCallback(() => {
    clientRef.current?.clearCache();
  }, []);

  return {
    translate,
    isTranslating,
    error,
    progress,
    downloadVocabulary,
    getVocabularyInfo,
    getSupportedLanguages,
    clearCache,
  };
}

// Export everything
export { TranslationEncoder };
export default TranslationClient;