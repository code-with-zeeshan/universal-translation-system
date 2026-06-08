// flutter/universal_translation_sdk/lib/src/translation_encoder.dart
import 'dart:async';
import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:logger/logger.dart';
import 'package:path/path.dart' as path;
import 'package:path_provider/path_provider.dart';

import 'models/vocabulary_pack.dart';
import 'vocabulary_manager.dart';

// FFI type definitions
typedef InitEncoderNative = Pointer<Void> Function(Pointer<Utf8> modelPath);
typedef InitEncoder = Pointer<Void> Function(Pointer<Utf8> modelPath);

typedef LoadVocabularyNative = Int32 Function(
  Pointer<Void> handle,
  Pointer<Utf8> vocabPath,
);
typedef LoadVocabulary = int Function(
  Pointer<Void> handle,
  Pointer<Utf8> vocabPath,
);

typedef EncodeNative = Pointer<Uint8> Function(
  Pointer<Void> handle,
  Pointer<Utf8> text,
  Pointer<Utf8> sourceLang,
  Pointer<Utf8> targetLang,
  Pointer<Int32> outSize,
);
typedef Encode = Pointer<Uint8> Function(
  Pointer<Void> handle,
  Pointer<Utf8> text,
  Pointer<Utf8> sourceLang,
  Pointer<Utf8> targetLang,
  Pointer<Int32> outSize,
);

typedef DestroyEncoderNative = Void Function(Pointer<Void> handle);
typedef DestroyEncoder = void Function(Pointer<Void> handle);

typedef GetMemoryUsageNative = Int64 Function(Pointer<Void> handle);
typedef GetMemoryUsage = int Function(Pointer<Void> handle);

/// Native translation encoder using FFI
class TranslationEncoder {
  static final Logger _logger = Logger();
  
  late final DynamicLibrary _lib;
  late final InitEncoder _initEncoder;
  late final LoadVocabulary _loadVocabulary;
  late final Encode _encode;
  late final DestroyEncoder _destroyEncoder;
  late final GetMemoryUsage _getMemoryUsage;
  
  Pointer<Void>? _handle;
  final VocabularyManager _vocabManager = VocabularyManager();
  bool _isInitialized = false;
  
  /// Initialize the translation encoder
  TranslationEncoder() {
    _loadLibrary();
  }
  
  /// Load native library based on platform
  void _loadLibrary() {
    try {
      if (Platform.isAndroid) {
        _lib = DynamicLibrary.open('libuniversal_encoder.so');
      } else if (Platform.isIOS) {
        _lib = DynamicLibrary.process();
      } else if (Platform.isMacOS) {
        _lib = DynamicLibrary.open('libuniversal_encoder.dylib');
      } else if (Platform.isLinux) {
        _lib = DynamicLibrary.open('libuniversal_encoder.so');
      } else if (Platform.isWindows) {
        _lib = DynamicLibrary.open('universal_encoder.dll');
      } else {
        throw UnsupportedError('Platform ${Platform.operatingSystem} not supported');
      }
      
      // Look up functions
      _initEncoder = _lib
          .lookup<NativeFunction<InitEncoderNative>>('init_encoder')
          .asFunction<InitEncoder>();
          
      _loadVocabulary = _lib
          .lookup<NativeFunction<LoadVocabularyNative>>('load_vocabulary')
          .asFunction<LoadVocabulary>();
          
      _encode = _lib
          .lookup<NativeFunction<EncodeNative>>('encode')
          .asFunction<Encode>();
          
      _destroyEncoder = _lib
          .lookup<NativeFunction<DestroyEncoderNative>>('destroy_encoder')
          .asFunction<DestroyEncoder>();
          
      _getMemoryUsage = _lib
          .lookup<NativeFunction<GetMemoryUsageNative>>('get_memory_usage')
          .asFunction<GetMemoryUsage>();
          
      _logger.i('Native library loaded successfully');
    } catch (e) {
      _logger.e('Failed to load native library', error: e);
      rethrow;
    }
  }
  
  /// Initialize the encoder with model
  Future<void> initialize() async {
    if (_isInitialized) return;
    
    try {
      // Get application documents directory
      final dir = await getApplicationDocumentsDirectory();
      final modelPath = path.join(dir.path, 'models', 'universal_encoder_int8.onnx');
      
      // Ensure directory exists
      final modelDir = Directory(path.dirname(modelPath));
      if (!await modelDir.exists()) {
        await modelDir.create(recursive: true);
      }
      
      // Extract model from assets if needed
      final modelFile = File(modelPath);
      if (!await modelFile.exists()) {
        _logger.i('Extracting model from assets...');
        final data = await rootBundle.load('assets/models/universal_encoder_int8.onnx');
        final bytes = data.buffer.asUint8List();
        await modelFile.writeAsBytes(bytes);
        _logger.i('Model extracted successfully: ${bytes.length} bytes');
      }
      
      // Initialize native encoder
      final pathPtr = modelPath.toNativeUtf8();
      try {
        _handle = _initEncoder(pathPtr);
        if (_handle == nullptr) {
          throw Exception('Failed to initialize native encoder');
        }
        _isInitialized = true;
        _logger.i('Encoder initialized successfully');
      } finally {
        malloc.free(pathPtr);
      }
    } catch (e) {
      _logger.e('Failed to initialize encoder', error: e);
      rethrow;
    }
  }
  
  /// Prepare for translation by loading appropriate vocabulary
  Future<void> prepareTranslation(String sourceLang, String targetLang) async {
    if (!_isInitialized) {
      await initialize();
    }
    
    try {
      // Get vocabulary pack for language pair
      final vocabPack = await _vocabManager.getVocabularyForPair(sourceLang, targetLang);
      
      // Download if needed
      if (vocabPack.needsDownload) {
        _logger.i('Downloading vocabulary pack: ${vocabPack.name}');
        await _downloadVocabulary(vocabPack);
      }
      
      // Load vocabulary into native encoder
      final vocabPathPtr = vocabPack.localPath.toNativeUtf8();
      try {
        final result = _loadVocabulary(_handle!, vocabPathPtr);
        if (result == 0) {
          throw Exception('Failed to load vocabulary');
        }
        _logger.i('Vocabulary loaded successfully: ${vocabPack.name}');
      } finally {
        malloc.free(vocabPathPtr);
      }
    } catch (e) {
      _logger.e('Failed to prepare translation', error: e);
      rethrow;
    }
  }
  
  /// Encode text using native encoder
  Future<Uint8List> encode(
    String text,
    String sourceLang,
    String targetLang,
  ) async {
    if (!_isInitialized) {
      await initialize();
    }
    
    if (_handle == null || _handle == nullptr) {
      throw StateError('Encoder not initialized');
    }
    
    // Prepare translation (load vocabulary if needed)
    await prepareTranslation(sourceLang, targetLang);
    
    // Convert strings to native format
    final textPtr = text.toNativeUtf8();
    final sourceLangPtr = sourceLang.toNativeUtf8();
    final targetLangPtr = targetLang.toNativeUtf8();
    final outSizePtr = malloc<Int32>();
    
    try {
      // Call native encode function
      final resultPtr = _encode(
        _handle!,
        textPtr,
        sourceLangPtr,
        targetLangPtr,
        outSizePtr,
      );
      
      if (resultPtr == nullptr) {
        throw Exception('Encoding failed');
      }
      
      // Get size and copy data
      final size = outSizePtr.value;
      final result = resultPtr.asTypedList(size);
      
      // Copy to Dart memory
      final bytes = Uint8List.fromList(result);
      
      // Free native result memory
      malloc.free(resultPtr);
      
      _logger.d('Encoded ${text.length} chars to ${bytes.length} bytes');
      
      return bytes;
    } finally {
      // Clean up
      malloc.free(textPtr);
      malloc.free(sourceLangPtr);
      malloc.free(targetLangPtr);
      malloc.free(outSizePtr);
    }
  }
  
  /// Download vocabulary pack — try HF Hub first, CDN fallback
  Future<void> _downloadVocabulary(VocabularyPack pack) async {
    final urls = [
      pack.downloadUrl,
      pack.downloadUrl.replaceFirst(
        'https://huggingface.co/your-org/universal-translation-system/resolve/main/vocabs',
        'https://cdn.yourdomain.com/vocabs',
      ),
    ];
    
    for (final url in urls) {
      try {
        final response = await http.get(Uri.parse(url));
        if (response.statusCode == 200) {
          final file = File(pack.localPath);
          await file.parent.create(recursive: true);
          await file.writeAsBytes(response.bodyBytes);
          _logger.i('Downloaded vocabulary pack: ${pack.name} from $url');
          return;
        }
        _logger.w('HTTP ${response.statusCode} from $url');
      } catch (e) {
        _logger.w('Failed to download from $url: $e');
      }
    }
    throw Exception('Failed to download vocabulary: all sources failed');
  }
  
  /// Get current memory usage
  int getMemoryUsage() {
    if (_handle == null || _handle == nullptr) {
      return 0;
    }
    return _getMemoryUsage(_handle!);
  }
  
  /// Dispose of native resources
  void dispose() {
    if (_handle != null && _handle != nullptr) {
      _destroyEncoder(_handle!);
      _handle = null;
      _isInitialized = false;
      _logger.i('Encoder disposed');
    }
  }
}

/// Translation result
sealed class TranslationResult {
  const TranslationResult();
}

class TranslationSuccess extends TranslationResult {
  final String translation;
  final double? confidence;
  
  const TranslationSuccess({
    required this.translation,
    this.confidence,
  });
}

class TranslationError extends TranslationResult {
  final String message;
  final String? code;
  
  const TranslationError({
    required this.message,
    this.code,
  });
}

// Coordinator status response model
class _CoordinatorStatus {
  final bool singleDecoder;
  final List<_CoordinatorDecoder> decoders;
  
  _CoordinatorStatus({required this.singleDecoder, required this.decoders});
  
  factory _CoordinatorStatus.fromJson(Map<String, dynamic> json) {
    return _CoordinatorStatus(
      singleDecoder: json['single_decoder'] as bool,
      decoders: (json['decoders'] as List)
          .map((d) => _CoordinatorDecoder.fromJson(d as Map<String, dynamic>))
          .toList(),
    );
  }
}

class _CoordinatorDecoder {
  final String nodeId;
  final String endpoint;
  
  _CoordinatorDecoder({required this.nodeId, required this.endpoint});
  
  factory _CoordinatorDecoder.fromJson(Map<String, dynamic> json) {
    return _CoordinatorDecoder(
      nodeId: json['node_id'] as String,
      endpoint: json['endpoint'] as String,
    );
  }
}

/// Translation client for end-to-end translation
class TranslationClient {
  static final Logger _logger = Logger();
  final TranslationEncoder _encoder = TranslationEncoder();
  final String decoderUrl;
  final String? coordinatorUrl;
  final String? localDecoderUrl;
  final bool preferLocal;
  final String hfRepo;
  final http.Client _httpClient = http.Client();
  final Duration timeout;
  
  String _effectiveDecoderUrl;
  bool _useCoordinator = false;
  bool _localDecoderAvailable = false;
  
  TranslationClient({
    this.decoderUrl = 'https://api.yourdomain.com/decode',
    this.coordinatorUrl,
    this.localDecoderUrl,
    this.preferLocal = true,
    this.hfRepo = 'your-org/universal-translation-system',
    this.timeout = const Duration(seconds: 30),
  }) : _effectiveDecoderUrl = decoderUrl;
  
  /// Initialize the client
  Future<void> initialize() async {
    await _encoder.initialize();
    if (localDecoderUrl != null && preferLocal) {
      await _checkLocalDecoder();
    }
    await _resolveCoordinator();
    await _checkEncoderUpdate();
  }

  Future<void> _checkLocalDecoder() async {
    if (localDecoderUrl == null) {
      _localDecoderAvailable = false;
      return;
    }
    try {
      final response = await _httpClient
          .get(Uri.parse('$localDecoderUrl/health'))
          .timeout(const Duration(seconds: 2));
      _localDecoderAvailable = response.statusCode == 200;
    } catch (e) {
      _logger.w('Local decoder unreachable: $e');
      _localDecoderAvailable = false;
    }
  }

  Future<void> _resolveCoordinator() async {
    if (coordinatorUrl == null) return;
    try {
      final response = await _httpClient
          .get(Uri.parse('$coordinatorUrl/api/status'))
          .timeout(const Duration(seconds: 5));
      if (response.statusCode == 200) {
        final status = _CoordinatorStatus.fromJson(jsonDecode(response.body));
        if (status.singleDecoder && status.decoders.isNotEmpty) {
          _effectiveDecoderUrl = status.decoders.first.endpoint;
          _useCoordinator = false;
        } else {
          _useCoordinator = true;
        }
      }
    } catch (e) {
      _logger.w('Coordinator unreachable, falling back to direct decoder: $e');
      _effectiveDecoderUrl = decoderUrl;
    }
  }

  String _getRequestUrl() {
    if (_localDecoderAvailable && preferLocal && localDecoderUrl != null) {
      return localDecoderUrl;
    }
    if (_useCoordinator && coordinatorUrl != null) {
      return '$coordinatorUrl/api/decode';
    }
    return _effectiveDecoderUrl;
  }

  Future<void> _checkEncoderUpdate() async {
    try {
      final response = await _httpClient
          .get(Uri.parse('https://huggingface.co/$hfRepo/raw/main/models/production/encoder.onnx'))
          .timeout(const Duration(seconds: 5));
      // HEAD not easily available in dart:http, so we check content-length and last-modified
      final remoteEtag = response.headers['etag'] ?? response.headers['last-modified'] ?? '';
      if (remoteEtag.isNotEmpty) {
        final cached = _getCachedEtag();
        if (remoteEtag != cached) {
          _cacheEtag(remoteEtag);
          await _downloadEncoderUpdate();
        }
      }
    } catch (e) {
      // Offline or HF unavailable, bundled encoder is fine
    }
  }

  String? _getCachedEtag() {
    // In production, this would read from SharedPreferences or secure storage
    return null;
  }

  void _cacheEtag(String etag) {
    // In production, this would persist the etag
  }

  Future<void> _downloadEncoderUpdate() async {
    try {
      final response = await _httpClient
          .get(Uri.parse('https://huggingface.co/$hfRepo/resolve/main/models/production/encoder.onnx'))
          .timeout(const Duration(seconds: 30));
      if (response.statusCode == 200) {
        // In production, this would update the encoder model file
        _logger.i('Encoder update downloaded (${response.bodyBytes.length} bytes)');
      }
    } catch (e) {
      _logger.w('Failed to download encoder update, using bundled version: $e');
    }
  }
  
  /// Translate text from source to target language
  Future<TranslationResult> translate({
    required String text,
    required String sourceLang,
    required String targetLang,
  }) async {
    try {
      if (text.trim().isEmpty) {
        return const TranslationError(
          message: 'Text cannot be empty',
          code: 'EMPTY_TEXT',
        );
      }
      
      _logger.d('Encoding text: ${text.length} characters');
      final encoded = await _encoder.encode(text, sourceLang, targetLang);
      
      _logger.d('Sending to decoder: ${encoded.length} bytes');
      final response = await _httpClient
          .post(
            Uri.parse(_getRequestUrl()),
            headers: {
              'Content-Type': 'application/octet-stream',
              'X-Target-Language': targetLang,
              'X-Source-Language': sourceLang,
            },
            body: encoded,
          )
          .timeout(timeout);
      
      if (response.statusCode == 200) {
        final result = jsonDecode(response.body);
        return TranslationSuccess(
          translation: result['translation'] as String,
          confidence: result['confidence'] as double?,
        );
      } else if (response.statusCode == 429) {
        return const TranslationError(
          message: 'Rate limit exceeded. Please try again later.',
          code: 'RATE_LIMIT',
        );
      } else {
        return TranslationError(
          message: 'Translation failed',
          code: 'HTTP_${response.statusCode}',
        );
      }
    } on TimeoutException {
      return const TranslationError(
        message: 'Request timed out',
        code: 'TIMEOUT',
      );
    } on SocketException {
      return const TranslationError(
        message: 'No internet connection',
        code: 'NO_INTERNET',
      );
    } catch (e) {
      _logger.e('Translation error', error: e);
      return TranslationError(
        message: e.toString(),
        code: 'UNKNOWN',
      );
    }
  }
  
  /// Get current memory usage
  int getMemoryUsage() {
    return _encoder.getMemoryUsage();
  }
  
  /// Dispose of resources
  void dispose() {
    _encoder.dispose();
    _httpClient.close();
  }
}