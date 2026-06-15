import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

final class TranslationResult {
  final String text;
  final String sourceLang;
  final String targetLang;
  final double confidence;

  const TranslationResult({
    required this.text,
    required this.sourceLang,
    required this.targetLang,
    required this.confidence,
  });
}

final class TranslationSuccess extends TranslationResult {
  const TranslationSuccess({
    required super.text,
    required super.sourceLang,
    required super.targetLang,
    required super.confidence,
  });
}

final class TranslationError extends TranslationResult {
  final String errorMessage;

  const TranslationError({
    required this.errorMessage,
    required super.text,
    required super.sourceLang,
    required super.targetLang,
    required super.confidence,
  });
}

typedef NativeInit = Pointer<Void> Function(Pointer<Utf8>);
typedef DartInit = Pointer<Void> Function(Pointer<Utf8>);

typedef NativeLoadVocab = Uint8 Function(Pointer<Void>, Pointer<Utf8>);
typedef DartLoadVocab = int Function(Pointer<Void>, Pointer<Utf8>);

typedef NativeEncode = Pointer<Uint8> Function(
  Pointer<Void>,
  Pointer<Utf8>,
  Pointer<Utf8>,
  Pointer<Utf8>,
  Pointer<Uint64>,
);
typedef DartEncode = Pointer<Uint8> Function(
  Pointer<Void>,
  Pointer<Utf8>,
  Pointer<Utf8>,
  Pointer<Utf8>,
  Pointer<Uint64>,
);

typedef NativeDestroy = Void Function(Pointer<Void>);
typedef DartDestroy = void Function(Pointer<Void>);

class TranslationClient {
  DynamicLibrary? _lib;
  Pointer<Void>? _encoder;

  bool get isInitialized => _encoder != null;

  TranslationClient();

  Future<void> initialize(String modelPath) async {
    if (_encoder != null) return;

    _lib = DynamicLibrary.open('libuniversal_encoder.so');

    final nativeInit = _lib!
        .lookupFunction<NativeInit, DartInit>('universal_encoder_init');
    final pathPtr = modelPath.toNativeUtf8();
    _encoder = nativeInit(pathPtr);
    calloc.free(pathPtr);
  }

  Future<void> loadVocabulary(String vocabPath) async {
    if (_encoder == null || _lib == null) {
      throw StateError('Encoder not initialized. Call initialize() first.');
    }

    final nativeLoad = _lib!
        .lookupFunction<NativeLoadVocab, DartLoadVocab>(
            'universal_encoder_load_vocabulary');
    final pathPtr = vocabPath.toNativeUtf8();
    final result = nativeLoad(_encoder!, pathPtr);
    calloc.free(pathPtr);

    if (result == 0) {
      throw Exception('Failed to load vocabulary from $vocabPath');
    }
  }

  Future<TranslationResult> translate({
    required String text,
    required String sourceLang,
    required String targetLang,
  }) async {
    if (_encoder == null || _lib == null) {
      throw StateError('Encoder not initialized. Call initialize() first.');
    }

    final nativeEncode = _lib!
        .lookupFunction<NativeEncode, DartEncode>('universal_encoder_encode');

    final textPtr = text.toNativeUtf8();
    final sourcePtr = sourceLang.toNativeUtf8();
    final targetPtr = targetLang.toNativeUtf8();
    final outLen = calloc<Uint64>();

    final resultPtr = nativeEncode(
      _encoder!,
      textPtr,
      sourcePtr,
      targetPtr,
      outLen,
    );

    final length = outLen.value;
    final data = Uint8List.fromList(
      resultPtr.asTypedList(length),
    );

    calloc.free(textPtr);
    calloc.free(sourcePtr);
    calloc.free(targetPtr);
    calloc.free(outLen);

    final decoded = utf8.decode(data);
    return TranslationSuccess(
      text: decoded,
      sourceLang: sourceLang,
      targetLang: targetLang,
      confidence: 1.0,
    );
  }

  List<String> getSupportedLanguages() {
    if (_encoder == null || _lib == null) return [];
    final nativeGetLangs =
        _lib!.lookupFunction<Pointer<Utf8> Function(Pointer<Void>),
            Pointer<Utf8> Function(Pointer<Void>)>(
                'universal_encoder_get_supported_languages');
    final resultPtr = nativeGetLangs(_encoder!);
    final result = resultPtr.toDartString();
    calloc.free(resultPtr);
    return result.split(',');
  }

  int getMemoryUsage() {
    if (_encoder == null || _lib == null) return 0;
    final nativeMem =
        _lib!.lookupFunction<Uint64 Function(Pointer<Void>),
            int Function(Pointer<Void>)>(
                'universal_encoder_get_memory_usage');
    return nativeMem(_encoder!);
  }

  void dispose() {
    if (_encoder != null && _lib != null) {
      final nativeDestroy = _lib!
          .lookupFunction<NativeDestroy, DartDestroy>(
              'universal_encoder_destroy');
      nativeDestroy(_encoder!);
      _encoder = null;
    }
    _lib = null;
  }
}
