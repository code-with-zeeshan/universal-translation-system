// flutter/universal_translation_sdk/lib/src/ui/translation_screen.dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../translation_encoder.dart';

/// Example translation screen
class TranslationScreen extends StatefulWidget {
  const TranslationScreen({super.key});

  @override
  State<TranslationScreen> createState() => _TranslationScreenState();
}

class _TranslationScreenState extends State<TranslationScreen> {
  final _translationClient = TranslationClient();
  final _inputController = TextEditingController();
  final _formKey = GlobalKey<FormState>();
  
  String _translatedText = '';
  bool _isTranslating = false;
  String _selectedSourceLang = 'en';
  String _selectedTargetLang = 'es';
  String? _errorMessage;
  
  final List<LanguageOption> _languages = [
    LanguageOption('en', 'English'),
    LanguageOption('es', 'Spanish'),
    LanguageOption('fr', 'French'),
    LanguageOption('de', 'German'),
    LanguageOption('zh', 'Chinese'),
    LanguageOption('ja', 'Japanese'),
    LanguageOption('ko', 'Korean'),
    LanguageOption('ar', 'Arabic'),
    LanguageOption('hi', 'Hindi'),
    LanguageOption('ru', 'Russian'),
    LanguageOption('pt', 'Portuguese'),
  ];
  
  @override
  void initState() {
    super.initState();
    _initializeClient();
  }
  
  Future<void> _initializeClient() async {
    try {
      await _translationClient.initialize();
    } catch (e) {
      if (mounted) {
        setState(() {
          _errorMessage = 'Failed to initialize: $e';
        });
      }
    }
  }
  
  Future<void> _translate() async {
    if (!_formKey.currentState!.validate()) return;
    
    setState(() {
      _isTranslating = true;
      _errorMessage = null;
    });
    
    try {
      final result = await _translationClient.translate(
        text: _inputController.text,
        from: _selectedSourceLang,
        to: _selectedTargetLang,
      );
      
      if (mounted) {
        setState(() {
          switch (result) {
            case TranslationSuccess(:final translation, :final confidence):
              _translatedText = translation;
              if (confidence != null) {
                _translatedText += '\n\nConfidence: ${(confidence * 100).toStringAsFixed(1)}%';
              }
            case TranslationError(:final message):
              _errorMessage = message;
              _translatedText = '';
          }
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _errorMessage = 'Translation error: $e';
          _translatedText = '';
        });
      }
    } finally {
      if (mounted) {
        setState(() => _isTranslating = false);
      }
    }
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Universal Translation'),
        elevation: 2,
      ),
      body: Form(
        key: _formKey,
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            // Language selection
            Row(
              children: [
                Expanded(
                  child: DropdownButtonFormField<String>(
                    value: _selectedSourceLang,
                    decoration: const InputDecoration(
                      labelText: 'From',
                      border: OutlineInputBorder(),
                    ),
                    items: _languages.map((lang) {
                      return DropdownMenuItem(
                        value: lang.code,
                        child: Text(lang.name),
                      );
                    }).toList(),
                    onChanged: (value) {
                      setState(() {
                        _selectedSourceLang = value!;
                      });
                    },
                  ),
                ),
                const SizedBox(width: 16),
                IconButton(
                  icon: const Icon(Icons.swap_horiz),
                  onPressed: () {
                    setState(() {
                      final temp = _selectedSourceLang;
                      _selectedSourceLang = _selectedTargetLang;
                      _selectedTargetLang = temp;
                    });
                  },
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: DropdownButtonFormField<String>(
                    value: _selectedTargetLang,
                    decoration: const InputDecoration(
                      labelText: 'To',
                      border: OutlineInputBorder(),
                    ),
                    items: _languages.map((lang) {
                      return DropdownMenuItem(
                        value: lang.code,
                        child: Text(lang.name),
                      );
                    }).toList(),
                    onChanged: (value) {
                      setState(() {
                        _selectedTargetLang = value!;
                      });
                    },
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            
            // Input field
            TextFormField(
              controller: _inputController,
              decoration: const InputDecoration(
                labelText: 'Enter text to translate',
                border: OutlineInputBorder(),
                alignLabelWithHint: true,
              ),
              maxLines: 5,
              maxLength: 500,
              validator: (value) {
                if (value == null || value.trim().isEmpty) {
                  return 'Please enter some text';
                }
                return null;
              },
            ),
            const SizedBox(height: 16),
            
            // Translate button
            SizedBox(
              height: 48,
              child: ElevatedButton.icon(
                onPressed: _isTranslating ? null : _translate,
                icon: _isTranslating
                    ? const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                        ),
                      )
                    : const Icon(Icons.translate),
                label: Text(_isTranslating ? 'Translating...' : 'Translate'),
              ),
            ),
            const SizedBox(height: 24),
            
            // Error message
            if (_errorMessage != null)
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.red.shade50,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.red.shade200),
                ),
                child: Row(
                  children: [
                    Icon(Icons.error_outline, color: Colors.red.shade700),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        _errorMessage!,
                        style: TextStyle(color: Colors.red.shade700),
                      ),
                    ),
                  ],
                ),
              ),
            
            // Translation result
            if (_translatedText.isNotEmpty && _errorMessage == null) ...[
              const Text(
                'Translation:',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 8),
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.blue.shade50,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.blue.shade200),
                ),
                child: SelectableText(
                  _translatedText,
                  style: const TextStyle(fontSize: 16),
                ),
              ),
            ],
            
            // Memory usage
            const SizedBox(height: 24),
            FutureBuilder<int>(
              future: Future.value(_translationClient.getMemoryUsage()),
              builder: (context, snapshot) {
                if (snapshot.hasData) {
                  final memoryMB = snapshot.data! / (1024 * 1024);
                  return Text(
                    'Memory usage: ${memoryMB.toStringAsFixed(1)} MB',
                    style: Theme.of(context).textTheme.bodySmall,
                  );
                }
                return const SizedBox.shrink();
              },
            ),
          ],
        ),
      ),
    );
  }
  
  @override
  void dispose() {
    _inputController.dispose();
    _translationClient.dispose();
    super.dispose();
  }
}

/// Language option
class LanguageOption {
  final String code;
  final String name;
  
  const LanguageOption(this.code, this.name);
}