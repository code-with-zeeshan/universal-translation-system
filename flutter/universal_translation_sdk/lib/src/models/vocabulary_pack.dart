// flutter/universal_translation_sdk/lib/src/models/vocabulary_pack.dart
import 'dart:io';
import 'package:json_annotation/json_annotation.dart';

part 'vocabulary_pack.g.dart';

/// Vocabulary pack information
@JsonSerializable()
class VocabularyPack {
  final String name;
  final List<String> languages;
  final String downloadUrl;
  final String localPath;
  final double sizeMb;
  final String version;
  
  const VocabularyPack({
    required this.name,
    required this.languages,
    required this.downloadUrl,
    required this.localPath,
    required this.sizeMb,
    required this.version,
  });
  
  /// Check if vocabulary needs to be downloaded
  bool get needsDownload => !File(localPath).existsSync();
  
  /// Check if vocabulary is downloaded
  bool get isDownloaded => File(localPath).existsSync();
  
  /// Get file size in bytes
  int get sizeBytes => (sizeMb * 1024 * 1024).round();
  
  factory VocabularyPack.fromJson(Map<String, dynamic> json) =>
      _$VocabularyPackFromJson(json);
      
  Map<String, dynamic> toJson() => _$VocabularyPackToJson(this);
}