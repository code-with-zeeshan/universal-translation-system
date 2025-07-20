// react-native/UniversalTranslationSDK/src/index.tsx

import {
  NativeModules,
  Platform,
  NativeEventEmitter,
  EmitterSubscription,
} from 'react-native';
import { useState, useCallback, useEffect, useRef } from 'react';

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
  languages: string[];
  downloadUrl: string;
  localPath: string;
  sizeMb: number;
  version: string;
  needsDownload: boolean;
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
  initialize(decoderUrl: string): Promise<void>;
  translate(text: string, sourceLang: string, targetLang: string): Promise<TranslationResult>;
  prepareTranslation(sourceLang: string, targetLang: string): Promise<void>;
  getVocabularyForPair(sourceLang: string, targetLang: string): Promise<VocabularyPack>;
  downloadVocabularyPacks(languages: string[]): Promise<void>;
  getSupportedLanguages(): Promise<LanguageInfo[]>;
  getMemoryUsage(): Promise<number>;
  clearTranslationCache(): Promise<void>;
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

// Event Emitter
const eventEmitter = new NativeEventEmitter(NativeModules.UniversalTranslationModule);

// Translation Client Class
export class TranslationClient {
  private initialized = false;
  private initPromise: Promise<void> | null = null;
  private decoderUrl: string;
  private cache: Map<string, TranslationResult>;
  private maxCacheSize: number;
  private preparedPairs: Set<string> = new Set();

  constructor(options?: {
    decoderUrl?: string;
    maxCacheSize?: number;
  }) {
    this.decoderUrl = options?.decoderUrl || 'https://api.yourdomain.com/decode';
    this.maxCacheSize = options?.maxCacheSize || 100;
    this.cache = new Map();
  }

  private async initialize(): Promise<void> {
    if (this.initialized) return;
    
    if (this.initPromise) return this.initPromise;

    this.initPromise = UniversalTranslationModule.initialize(this.decoderUrl)
      .then(() => {
        this.initialized = true;
      })
      .catch((error) => {
        this.initPromise = null;
        throw this.createError(error);
      });

    return this.initPromise;
  }

  async translate(options: TranslationOptions): Promise<TranslationResult> {
    await this.initialize();

    // Validate input
    if (!options.text?.trim()) {
      throw this.createError({ 
        code: 'INVALID_INPUT', 
        message: 'Text cannot be empty' 
      });
    }

    // Check cache
    const cacheKey = `${options.sourceLang}-${options.targetLang}:${options.text}`;
    const cached = this.cache.get(cacheKey);
    if (cached) {
      return cached;
    }

    try {
      // Prepare translation if not already done
      const pairKey = `${options.sourceLang}-${options.targetLang}`;
      if (!this.preparedPairs.has(pairKey)) {
        await this.prepareLanguagePair(options.sourceLang, options.targetLang);
        this.preparedPairs.add(pairKey);
      }

      // Translate using native module
      const result = await UniversalTranslationModule.translate(
        options.text,
        options.sourceLang,
        options.targetLang
      );

      // Add to cache
      this.addToCache(cacheKey, result);

      return result;
    } catch (error) {
      throw this.createError(error);
    }
  }

  async translateBatch(
    texts: string[],
    sourceLang: string,
    targetLang: string
  ): Promise<TranslationResult[]> {
    // Prepare language pair once
    await this.prepareLanguagePair(sourceLang, targetLang);

    // Translate all texts
    const promises = texts.map((text) =>
      this.translate({ text, sourceLang, targetLang })
    );
    
    return Promise.all(promises);
  }

  async prepareLanguagePair(sourceLang: string, targetLang: string): Promise<void> {
    await this.initialize();
    
    try {
      await UniversalTranslationModule.prepareTranslation(sourceLang, targetLang);
    } catch (error) {
      throw this.createError(error);
    }
  }

  async getVocabularyInfo(sourceLang: string, targetLang: string): Promise<VocabularyPack> {
    await this.initialize();
    
    try {
      return await UniversalTranslationModule.getVocabularyForPair(sourceLang, targetLang);
    } catch (error) {
      throw this.createError(error);
    }
  }

  async downloadVocabulariesForLanguages(languages: string[]): Promise<void> {
    await this.initialize();
    
    try {
      await UniversalTranslationModule.downloadVocabularyPacks(languages);
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
    await this.initialize();
    
    try {
      return await UniversalTranslationModule.getMemoryUsage();
    } catch (error) {
      throw this.createError(error);
    }
  }

  clearCache(): void {
    this.cache.clear();
    this.preparedPairs.clear();
  }

  async clearAllCaches(): Promise<void> {
    this.clearCache();
    
    try {
      await UniversalTranslationModule.clearTranslationCache();
    } catch (error) {
      // Ignore cache clear errors
      console.warn('Failed to clear native cache:', error);
    }
  }

  subscribeToDownloadProgress(callback: (progress: { language: string; progress: number }) => void): EmitterSubscription {
    return eventEmitter.addListener('vocabularyDownloadProgress', callback);
  }

  private addToCache(key: string, result: TranslationResult): void {
    // Implement LRU cache
    if (this.cache.size >= this.maxCacheSize) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
    this.cache.set(key, result);
  }

  private createError(error: any): TranslationError {
    const translationError = new Error(error.message || 'Unknown error') as TranslationError;
    translationError.code = error.code || 'UNKNOWN_ERROR';
    translationError.userMessage = this.getUserMessage(error.code, error.message);
    return translationError;
  }

  private getUserMessage(code: string, defaultMessage: string): string {
    const userMessages: Record<string, string> = {
      'INIT_ERROR': 'Failed to initialize translation service',
      'TRANSLATION_ERROR': 'Translation failed. Please try again.',
      'PREPARE_ERROR': 'Failed to prepare translation. Please check your internet connection.',
      'VOCAB_ERROR': 'Failed to load vocabulary pack',
      'DOWNLOAD_ERROR': 'Failed to download vocabulary. Please check your internet connection.',
      'INVALID_INPUT': 'Please enter valid text to translate',
      'NETWORK_ERROR': 'Network error. Please check your internet connection.',
    };

    return userMessages[code] || defaultMessage || 'An error occurred';
  }
}

// React Hook
export function useTranslation(options?: {
  decoderUrl?: string;
}) {
  const clientRef = useRef<TranslationClient>();
  const [isTranslating, setIsTranslating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [downloadProgress, setDownloadProgress] = useState<Record<string, number>>({});
  const progressSubscriptionRef = useRef<EmitterSubscription>();

  useEffect(() => {
    // Initialize client
    clientRef.current = new TranslationClient(options);

    // Subscribe to download progress
    progressSubscriptionRef.current = clientRef.current.subscribeToDownloadProgress(
      ({ language, progress }) => {
        setDownloadProgress((prev) => ({ ...prev, [language]: progress }));
      }
    );

    // Cleanup
    return () => {
      progressSubscriptionRef.current?.remove();
      clientRef.current?.clearCache();
    };
  }, [options?.decoderUrl]);

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

  const translateBatch = useCallback(async (
    texts: string[],
    sourceLang: string,
    targetLang: string
  ) => {
    if (!clientRef.current) {
      throw new Error('Translation client not initialized');
    }

    setIsTranslating(true);
    setError(null);

    try {
      const results = await clientRef.current.translateBatch(texts, sourceLang, targetLang);
      return results;
    } catch (err: any) {
      const errorMessage = err.userMessage || err.message || 'Translation failed';
      setError(errorMessage);
      throw err;
    } finally {
      setIsTranslating(false);
    }
  }, []);

  const getSupportedLanguages = useCallback(async () => {
    if (!clientRef.current) {
      throw new Error('Translation client not initialized');
    }

    return clientRef.current.getSupportedLanguages();
  }, []);

  const downloadLanguages = useCallback(async (languages: string[]) => {
    if (!clientRef.current) {
      throw new Error('Translation client not initialized');
    }

    try {
      await clientRef.current.downloadVocabulariesForLanguages(languages);
    } catch (err: any) {
      const errorMessage = err.userMessage || err.message || 'Download failed';
      setError(errorMessage);
      throw err;
    }
  }, []);

  const clearCache = useCallback(() => {
    clientRef.current?.clearCache();
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    translate,
    translateBatch,
    isTranslating,
    error,
    clearError,
    downloadProgress,
    downloadLanguages,
    getSupportedLanguages,
    clearCache,
  };
}

// Export everything
export default TranslationClient;
export { LanguageInfo, VocabularyPack, TranslationOptions, TranslationResult };