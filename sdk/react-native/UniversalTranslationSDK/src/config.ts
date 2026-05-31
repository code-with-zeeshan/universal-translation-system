/**
 * Configuration module for the Universal Translation SDK (React Native)
 * Centralizes all environment variables and default values
 */

export interface SdkConfig {
  // API endpoints
  decoderApiUrl: string;
  encoderApiUrl: string;
  
  // Model paths
  modelUrl: string;
  vocabUrl: string;
  
  // Version information
  modelVersion: string;
  
  // Feature flags
  useNativeEncoder: boolean;
  enableFallback: boolean;
}

/**
 * Get configuration from environment variables with fallbacks
 */
export function getConfig(): SdkConfig {
  return {
    // API endpoints
    decoderApiUrl: process.env.DECODER_API_URL || 'https://api.yourdomain.com/decode',
    encoderApiUrl: process.env.ENCODER_API_URL || 'https://api.universal-translation.com/encode',
    
    // Model paths
    modelUrl: process.env.MODEL_URL || '/models/universal_encoder.onnx',
    vocabUrl: process.env.VOCAB_URL || '/vocabs',
    
    // Version information
    modelVersion: process.env.MODEL_VERSION || '1.0.0',
    
    // Feature flags
    useNativeEncoder: process.env.USE_NATIVE_ENCODER !== 'false',
    enableFallback: process.env.ENABLE_FALLBACK !== 'false',
  };
}

// Export a singleton instance for easy imports
export const config = getConfig();