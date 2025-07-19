// flutter/universal_translation_sdk/lib/src/vocabulary_manager.dart
import 'dart:async';
import 'dart:io';

import 'package:logger/logger.dart';
import 'package:path/path.dart' as path;
import 'package:path_provider/path_provider.dart';

import 'models/vocabulary_pack.dart';

/// Manages vocabulary packs for different language pairs
class VocabularyManager {
  static final Logger _logger = Logger();
  
  static const Map<String, String> _languageToPack = {
    'en': 'latin',
    'es': 'latin',
    'fr': 'latin',
    'de': 'latin',
    'it': 'latin',
    'pt': 'latin',
    'nl': 'latin',
    'sv': 'latin',
    'zh': 'cjk',
    'ja': 'cjk',
    'ko': 'cjk',
    'ar': 'arabic',
    'hi': 'devanagari',
    'ru': 'cyrillic',
    'uk': 'cyrillic',
    'th': 'thai',
  };
  
  static const Map<String, List<String>> _packLanguages = {
    'latin': ['en', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'sv'],
    'cjk': ['zh', 'ja', 'ko'],
    'arabic': ['ar'],
    'devanagari': ['hi'],
    'cyrillic': ['ru', 'uk'],
    'thai': ['th'],
  };
  
  static const Map<String, double> _packSizes = {
    'latin': 5.0,
    'cjk': 8.0,
    'arabic': 3.0,
    'devanagari': 3.0,
    'cyrillic': 4.0,
    'thai': 2.0,
  };
  
  late final Directory _vocabDir;
  bool _initialized = false;
  
  /// Initialize vocabulary manager
  Future<void> _ensureInitialized() async {
    if (_initialized) return;
    
    final appDir = await getApplicationDocumentsDirectory();
    _vocabDir = Directory(path.join(appDir.path, 'vocabularies'));
    
    if (!await _vocabDir.exists()) {
      await _vocabDir.create(recursive: true);
    }
    
    _initialized = true;
  }
  
  /// Get vocabulary pack for a language pair
  Future<VocabularyPack> getVocabularyForPair(
    String sourceLang,
    String targetLang,
  ) async {
    await _ensureInitialized();
    
    // Determine which pack to use
    final sourcePack = _languageToPack[sourceLang] ?? 'latin';
    final targetPack = _languageToPack[targetLang] ?? 'latin';
    
    // Prioritize target language pack
    final packName = targetPack != 'latin' ? targetPack : sourcePack;
    
    final pack = VocabularyPack(
      name: packName,
      languages: _packLanguages[packName] ?? [],
      downloadUrl: _getDownloadUrl(packName),
      localPath: path.join(_vocabDir.path, '$packName.msgpack'),
      sizeMb: _packSizes[packName] ?? 5.0,
      version: '1.0',
    );
    
    _logger.i('Selected vocabulary pack: $packName for $sourceLang->$targetLang');
    
    return pack;
  }
  
  /// Get download URL for vocabulary pack
  String _getDownloadUrl(String packName) {
    // Replace with your actual CDN URL
    return 'https://cdn.yourdomain.com/vocabs/${packName}_v1.0.msgpack';
  }
  
  /// Get all downloaded packs
  Future<List<VocabularyPack>> getDownloadedPacks() async {
    await _ensureInitialized();
    
    final packs = <VocabularyPack>[];
    
    await for (final file in _vocabDir.list()) {
      if (file is File && file.path.endsWith('.msgpack')) {
        final packName = path.basenameWithoutExtension(file.path);
        if (_packLanguages.containsKey(packName)) {
          final pack = VocabularyPack(
            name: packName,
            languages: _packLanguages[packName]!,
            downloadUrl: _getDownloadUrl(packName),
            localPath: file.path,
            sizeMb: _packSizes[packName] ?? 5.0,
            version: '1.0',
          );
          packs.add(pack);
        }
      }
    }
    
    return packs;
  }
  
  /// Delete a vocabulary pack
  Future<void> deletePack(String packName) async {
    await _ensureInitialized();
    
    final packFile = File(path.join(_vocabDir.path, '$packName.msgpack'));
    if (await packFile.exists()) {
      await packFile.delete();
      _logger.i('Deleted vocabulary pack: $packName');
    }
  }
  
  /// Get total size of downloaded packs
  Future<int> getTotalSize() async {
    await _ensureInitialized();
    
    int totalSize = 0;
    
    await for (final file in _vocabDir.list()) {
      if (file is File) {
        totalSize += await file.length();
      }
    }
    
    return totalSize;
  }
}