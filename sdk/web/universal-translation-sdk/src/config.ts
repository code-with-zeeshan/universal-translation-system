/**
 * Configuration module for the Universal Translation SDK
 * Centralizes all environment variables and default values
 */

export interface SdkConfig {
  // API endpoints
  decoderApiUrl: string;
  coordinatorApiUrl: string;
  encoderApiUrl: string;
  
  // HF repo for encoder updates
  hfRepo: string;
  
  // Model paths
  modelUrl: string;
  vocabUrl: string;
  hfVocabUrl: string;
  wasmEncoderPath: string;
  
  // Version information
  modelVersion: string;
  
  // Feature flags
  useWasmEncoder: boolean;
  enableFallback: boolean;
}

/**
 * Get configuration from environment variables with fallbacks
 */
export function getConfig(): SdkConfig {
  return {
    // API endpoints
    decoderApiUrl: process.env.DECODER_API_URL || 'https://api.yourdomain.com/decode',
    coordinatorApiUrl: process.env.COORDINATOR_API_URL || '',
    encoderApiUrl: process.env.ENCODER_API_URL || 'https://api.universal-translation.com/encode',
    
    // HF repo for encoder update checks
    hfRepo: process.env.HF_REPO || 'your-org/universal-translation-system',
    
    // Model paths
    modelUrl: process.env.MODEL_URL || '/models/universal_encoder.onnx',
    vocabUrl: process.env.VOCAB_URL || '/vocabs',
    hfVocabUrl: process.env.HF_VOCAB_URL || 'https://huggingface.co/your-org/universal-translation-system/resolve/main/vocabs',
    wasmEncoderPath: process.env.WASM_ENCODER_PATH || '/wasm/encoder.js',
    
    // Version information
    modelVersion: process.env.MODEL_VERSION || '1.0.0',
    
    // Feature flags
    useWasmEncoder: process.env.USE_WASM_ENCODER !== 'false',
    enableFallback: process.env.ENABLE_FALLBACK !== 'false',
  };
}

// Export a singleton instance for easy imports
export const config = getConfig();