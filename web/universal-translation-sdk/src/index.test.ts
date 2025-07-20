// src/index.test.ts
import { TranslationEncoder, TranslationClient } from './index';

// Mock fetch for tests
global.fetch = jest.fn();

// Mock ONNX Runtime
jest.mock('onnxruntime-web', () => ({
  InferenceSession: {
    create: jest.fn().mockResolvedValue({
      inputNames: ['input_ids', 'attention_mask'],
      outputNames: ['encoder_output'],
      run: jest.fn().mockResolvedValue({
        encoder_output: {
          data: new Float32Array(128 * 1024).fill(0.1),
          dims: [1, 128, 1024]
        }
      })
    })
  },
  Tensor: jest.fn((type, data, dims) => ({ type, data, dims })),
  env: {
    wasm: { wasmPaths: '' },
    webgl: { powerPreference: '' }
  }
}));

describe('TranslationEncoder', () => {
  let encoder: TranslationEncoder;

  beforeEach(() => {
    encoder = new TranslationEncoder({
      modelUrl: '/test/model.onnx',
      vocabUrl: '/test/vocabs'
    });
    jest.clearAllMocks();
  });

  test('should initialize without errors', async () => {
    await expect(encoder.initialize()).resolves.not.toThrow();
  });

  test('should validate supported languages', () => {
    const languages = encoder.getSupportedLanguages();
    expect(languages).toContain('en');
    expect(languages).toContain('es');
    expect(languages).toContain('zh');
    expect(languages.length).toBe(20);
  });

  test('should handle unsupported languages', async () => {
    await expect(
      encoder.prepareTranslation('xyz', 'es')
    ).rejects.toThrow('Unsupported source language: xyz');
  });

  test('should load vocabulary pack', async () => {
    const mockVocab = {
      name: 'latin',
      version: '1.0',
      languages: ['en', 'es', 'fr'],
      tokens: { 'hello': 100, 'world': 101 },
      subwords: { '##ing': 200 },
      special_tokens: { '<s>': 2, '</s>': 3, '<pad>': 0, '<unk>': 1 },
      metadata: { total_tokens: 1000, size_mb: 5.0 }
    };

    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => mockVocab
    });

    await encoder.prepareTranslation('en', 'es');
    expect(fetch).toHaveBeenCalledWith('/test/vocabs/latin_v1.0.json');
  });

  test('should tokenize text correctly', async () => {
    const mockVocab = {
      name: 'latin',
      version: '1.0',
      languages: ['en'],
      tokens: { 'hello': 100, 'world': 101 },
      subwords: { '##unk': 200 },
      special_tokens: { 
        '<s>': 2, 
        '</s>': 3, 
        '<pad>': 0, 
        '<unk>': 1,
        '<en>': 10 
      },
      metadata: { total_tokens: 1000, size_mb: 5.0 }
    };

    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => mockVocab
    });

    await encoder.prepareTranslation('en', 'es');
    
    // Test tokenization through encode
    const result = await encoder.encode('hello world', 'en', 'es');
    expect(result).toBeInstanceOf(Uint8Array);
    expect(result.length).toBeGreaterThan(16); // At least header + data
  });

  test('should compress output correctly', async () => {
    await encoder.initialize();
    
    const mockVocab = {
      name: 'latin',
      version: '1.0',
      languages: ['en'],
      tokens: { 'test': 100 },
      subwords: {},
      special_tokens: { '<s>': 2, '</s>': 3, '<pad>': 0, '<unk>': 1 },
      metadata: { total_tokens: 100, size_mb: 1.0 }
    };

    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => mockVocab
    });

    const compressed = await encoder.encode('test', 'en', 'es');
    
    // Check header structure
    const view = new DataView(compressed.buffer, 0, 16);
    expect(view.getInt32(0, true)).toBe(128); // sequence length
    expect(view.getInt32(4, true)).toBe(1024); // hidden dim
    expect(view.getFloat32(8, true)).toBeGreaterThan(0); // scale
    expect(view.getInt32(12, true)).toBe(0); // compression flag
  });
});

describe('TranslationClient', () => {
  let client: TranslationClient;

  beforeEach(() => {
    client = new TranslationClient({
      modelUrl: '/test/model.onnx',
      decoderUrl: 'http://localhost:8000/decode'
    });
    jest.clearAllMocks();
  });

  test('should validate empty input', async () => {
    await expect(client.translate({
      text: '',
      sourceLang: 'en',
      targetLang: 'es'
    })).rejects.toThrow('Text cannot be empty');
  });

  test('should validate whitespace-only input', async () => {
    await expect(client.translate({
      text: '   \n\t  ',
      sourceLang: 'en',
      targetLang: 'es'
    })).rejects.toThrow('Text cannot be empty');
  });

  test('should cache translations', async () => {
    const mockVocab = {
      name: 'latin',
      version: '1.0',
      languages: ['en', 'es'],
      tokens: { 'hello': 100 },
      subwords: {},
      special_tokens: { '<s>': 2, '</s>': 3, '<pad>': 0, '<unk>': 1 },
      metadata: { total_tokens: 100, size_mb: 1.0 }
    };

    (global.fetch as jest.Mock)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockVocab
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ translation: 'hola', target_lang: 'es' })
      });

    await client.initialize();

    const options = {
      text: 'hello',
      sourceLang: 'en',
      targetLang: 'es'
    };

    // First call should fetch
    const result1 = await client.translate(options);
    expect(result1.translation).toBe('hola');

    // Second call should use cache
    const result2 = await client.translate(options);
    expect(result2.translation).toBe('hola');
    
    // Fetch should only be called twice (vocab + translation), not three times
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  test('should handle rate limiting', async () => {
    (global.fetch as jest.Mock)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          name: 'latin',
          version: '1.0',
          languages: ['en'],
          tokens: {},
          subwords: {},
          special_tokens: { '<s>': 2, '</s>': 3, '<pad>': 0, '<unk>': 1 },
          metadata: { total_tokens: 100, size_mb: 1.0 }
        })
      })
      .mockResolvedValueOnce({
        ok: false,
        status: 429,
        json: async () => ({ error: 'Rate limit exceeded' })
      });

    await client.initialize();

    await expect(client.translate({
      text: 'test',
      sourceLang: 'en',
      targetLang: 'es'
    })).rejects.toThrow('Rate limit exceeded');
  });

  test('should handle server errors', async () => {
    (global.fetch as jest.Mock)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          name: 'latin',
          version: '1.0',
          languages: ['en'],
          tokens: {},
          subwords: {},
          special_tokens: { '<s>': 2, '</s>': 3, '<pad>': 0, '<unk>': 1 },
          metadata: { total_tokens: 100, size_mb: 1.0 }
        })
      })
      .mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: async () => ({ error: 'Internal server error' })
      });

    await client.initialize();

    await expect(client.translate({
      text: 'test',
      sourceLang: 'en',
      targetLang: 'es'
    })).rejects.toThrow('Translation service is temporarily unavailable');
  });

  test('should translate batch', async () => {
    const mockVocab = {
      name: 'latin',
      version: '1.0',
      languages: ['en', 'es'],
      tokens: { 'hello': 100, 'world': 101 },
      subwords: {},
      special_tokens: { '<s>': 2, '</s>': 3, '<pad>': 0, '<unk>': 1 },
      metadata: { total_tokens: 100, size_mb: 1.0 }
    };

    (global.fetch as jest.Mock)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockVocab
      })
      .mockResolvedValue({
        ok: true,
        json: async () => ({ translation: 'translated', target_lang: 'es' })
      });

    await client.initialize();

    const results = await client.translateBatch(
      ['hello', 'world'],
      'en',
      'es'
    );

    expect(results).toHaveLength(2);
    expect(results[0].translation).toBe('translated');
    expect(results[1].translation).toBe('translated');
  });
});