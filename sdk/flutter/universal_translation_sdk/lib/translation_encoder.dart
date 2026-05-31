import 'package:opentelemetry_api/opentelemetry_api.dart';
final tracer = TracerProvider().getTracer('universal_translation_sdk');
const String modelVersion = String.fromEnvironment('MODEL_VERSION', defaultValue: '1.0.0');

class TranslationEncoder {
  // ... existing fields ...
  String get modelVersion => modelVersion;

  Future<void> initialize() async {
    final span = tracer.startSpan('TranslationEncoder.initialize');
    span.setAttribute('model_version', modelVersion);
    // ... existing logic ...
    span.end();
  }

  Future<void> loadVocabulary(String source, String target) async {
    final span = tracer.startSpan('TranslationEncoder.loadVocabulary');
    span.setAttribute('model_version', modelVersion);
    // ... existing logic ...
    span.end();
  }

  Future<List<double>> encode({required String text, required String sourceLang, required String targetLang}) async {
    final span = tracer.startSpan('TranslationEncoder.encode');
    span.setAttribute('model_version', modelVersion);
    // ... existing logic ...
    // When making HTTP requests, add trace context and model version header
    // Example: headers['X-Model-Version'] = modelVersion;
    span.end();
    return [];
  }
} 