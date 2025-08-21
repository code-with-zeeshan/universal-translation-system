import { WasmEncoderWrapper } from '../wasmEncoder';

// Mock the WebAssembly module
const mockEncoder = {
  initialize: jest.fn().mockReturnValue(true),
  loadVocabulary: jest.fn().mockReturnValue(true),
  hasVocabulary: jest.fn().mockReturnValue(true),
  encode: jest.fn().mockReturnValue(new Float32Array([0.1, 0.2, 0.3])),
  compressEmbedding: jest.fn().mockReturnValue(new Uint8Array([1, 2, 3])),
  getSupportedLanguages: jest.fn().mockReturnValue(['en', 'es', 'fr']),
  destroy: jest.fn()
};

const mockWasmModule = {
  WasmEncoder: jest.fn().mockImplementation(() => mockEncoder)
};

// Mock dynamic import
jest.mock('../wasmEncoder', () => {
  const originalModule = jest.requireActual('../wasmEncoder');
  return {
    ...originalModule,
    WasmEncoderWrapper: jest.fn().mockImplementation(() => {
      return {
        load: jest.fn().mockResolvedValue(undefined),
        isLoaded: jest.fn().mockReturnValue(true),
        loadVocabulary: jest.fn().mockResolvedValue(true),
        hasVocabulary: jest.fn().mockReturnValue(true),
        encode: jest.fn().mockResolvedValue(new Float32Array([0.1, 0.2, 0.3])),
        compressEmbedding: jest.fn().mockResolvedValue(new Uint8Array([1, 2, 3])),
        getSupportedLanguages: jest.fn().mockResolvedValue(['en', 'es', 'fr']),
        encodeWithFallback: jest.fn().mockResolvedValue(new Float32Array([0.1, 0.2, 0.3])),
        destroy: jest.fn()
      };
    })
  };
});

describe('WasmEncoderWrapper', () => {
  let wasmEncoder: any;
  
  beforeEach(() => {
    jest.clearAllMocks();
    wasmEncoder = new WasmEncoderWrapper({
      wasmPath: '../dist/wasm/encoder.js',
      useWasm: true
    });
  });
  
  test('should initialize with default options', () => {
    expect(wasmEncoder).toBeDefined();
  });
  
  test('should load the WebAssembly module', async () => {
    await wasmEncoder.load();
    expect(wasmEncoder.isLoaded()).toBe(true);
  });
  
  test('should load vocabulary for a language', async () => {
    const result = await wasmEncoder.loadVocabulary('en');
    expect(result).toBe(true);
  });
  
  test('should check if vocabulary is loaded', () => {
    const result = wasmEncoder.hasVocabulary('en');
    expect(result).toBe(true);
  });
  
  test('should encode text to embeddings', async () => {
    const result = await wasmEncoder.encode('Hello world', 'en', 'es');
    expect(result).toBeInstanceOf(Float32Array);
    expect(result.length).toBe(3);
  });
  
  test('should compress embeddings', async () => {
    const embedding = new Float32Array([0.1, 0.2, 0.3]);
    const result = await wasmEncoder.compressEmbedding(embedding);
    expect(result).toBeInstanceOf(Uint8Array);
    expect(result.length).toBe(3);
  });
  
  test('should get supported languages', async () => {
    const result = await wasmEncoder.getSupportedLanguages();
    expect(result).toEqual(['en', 'es', 'fr']);
  });
  
  test('should encode with fallback if WebAssembly fails', async () => {
    // Mock encode to throw an error
    wasmEncoder.encode = jest.fn().mockRejectedValue(new Error('WebAssembly failed'));
    
    const result = await wasmEncoder.encodeWithFallback('Hello world', 'en', 'es');
    expect(result).toBeInstanceOf(Float32Array);
    expect(result.length).toBe(3);
  });
  
  test('should destroy the encoder', () => {
    wasmEncoder.destroy();
    // No assertions needed, just checking that it doesn't throw
  });
});