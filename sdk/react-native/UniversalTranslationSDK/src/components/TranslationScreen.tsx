// react-native/UniversalTranslationSDK/src/components/TranslationScreen.tsx

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  ActivityIndicator,
  Alert,
  KeyboardAvoidingView,
  Platform,
  Modal,
  FlatList,
} from 'react-native';
import { useTranslation, LanguageInfo } from '../index';

interface TranslationScreenProps {
  decoderUrl?: string;
  defaultSourceLang?: string;
  defaultTargetLang?: string;
}

export function TranslationScreen({ 
  decoderUrl,
  defaultSourceLang = 'en',
  defaultTargetLang = 'es'
}: TranslationScreenProps) {
  const {
    translate,
    isTranslating,
    error,
    clearError,
    getSupportedLanguages,
    downloadLanguages,
    downloadProgress,
    clearCache,
  } = useTranslation({ decoderUrl });

  const [inputText, setInputText] = useState('');
  const [translatedText, setTranslatedText] = useState('');
  const [sourceLang, setSourceLang] = useState(defaultSourceLang);
  const [targetLang, setTargetLang] = useState(defaultTargetLang);
  const [languages, setLanguages] = useState<LanguageInfo[]>([]);
  const [showSourcePicker, setShowSourcePicker] = useState(false);
  const [showTargetPicker, setShowTargetPicker] = useState(false);
  const [isLoadingLanguages, setIsLoadingLanguages] = useState(true);

  useEffect(() => {
    loadLanguages();
  }, []);

  useEffect(() => {
    if (error) {
      Alert.alert('Error', error, [
        { text: 'OK', onPress: clearError }
      ]);
    }
  }, [error, clearError]);

  const loadLanguages = async () => {
    try {
      setIsLoadingLanguages(true);
      const langs = await getSupportedLanguages();
      setLanguages(langs);
      
      // Pre-download vocabularies for default languages
      await downloadLanguages([defaultSourceLang, defaultTargetLang]);
    } catch (err) {
      console.error('Failed to load languages:', err);
      Alert.alert('Error', 'Failed to load languages. Please restart the app.');
    } finally {
      setIsLoadingLanguages(false);
    }
  };

  const handleTranslate = async () => {
    if (!inputText.trim()) {
      Alert.alert('Error', 'Please enter text to translate');
      return;
    }

    try {
      const result = await translate({
        text: inputText,
        sourceLang,
        targetLang,
      });
      setTranslatedText(result.translation);
    } catch (err: any) {
      // Error is already handled by the hook
      console.error('Translation error:', err);
    }
  };

  const swapLanguages = async () => {
    const newSource = targetLang;
    const newTarget = sourceLang;
    
    setSourceLang(newSource);
    setTargetLang(newTarget);
    
    if (translatedText) {
      setInputText(translatedText);
      setTranslatedText('');
    }
    
    // Pre-download vocabularies for swapped languages
    try {
      await downloadLanguages([newSource, newTarget]);
    } catch (err) {
      console.error('Failed to prepare languages:', err);
    }
  };

  const getLanguageName = (code: string) => {
    return languages.find((lang) => lang.code === code)?.name || code;
  };

  const handleLanguageSelect = async (
    language: string, 
    isSource: boolean
  ) => {
    if (isSource) {
      setSourceLang(language);
      setShowSourcePicker(false);
    } else {
      setTargetLang(language);
      setShowTargetPicker(false);
    }
    
    // Download vocabulary for the new language
    try {
      await downloadLanguages([language]);
    } catch (err) {
      console.error('Failed to download vocabulary:', err);
    }
  };

  if (isLoadingLanguages) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#007AFF" />
        <Text style={styles.loadingText}>Loading languages...</Text>
      </View>
    );
  }

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* Language Selector */}
        <View style={styles.languageSelector}>
          <TouchableOpacity
            style={styles.languageButton}
            onPress={() => setShowSourcePicker(true)}
          >
            <Text style={styles.languageLabel}>From</Text>
            <Text style={styles.languageName}>{getLanguageName(sourceLang)}</Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.swapButton} onPress={swapLanguages}>
            <Text style={styles.swapIcon}>⇄</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.languageButton}
            onPress={() => setShowTargetPicker(true)}
          >
            <Text style={styles.languageLabel}>To</Text>
            <Text style={styles.languageName}>{getLanguageName(targetLang)}</Text>
          </TouchableOpacity>
        </View>

        {/* Download Progress */}
        {Object.keys(downloadProgress).length > 0 && (
          <View style={styles.progressContainer}>
            {Object.entries(downloadProgress).map(([lang, progress]) => (
              <View key={lang} style={styles.progressItem}>
                <Text style={styles.progressText}>
                  Downloading {getLanguageName(lang)}: {Math.round(progress)}%
                </Text>
                <View style={styles.progressBar}>
                  <View 
                    style={[styles.progressFill, { width: `${progress}%` }]} 
                  />
                </View>
              </View>
            ))}
          </View>
        )}

        {/* Input Section */}
        <View style={styles.inputSection}>
          <Text style={styles.sectionLabel}>Enter text</Text>
          <TextInput
            style={styles.input}
            value={inputText}
            onChangeText={setInputText}
            placeholder="Type or paste text here..."
            multiline
            numberOfLines={5}
            textAlignVertical="top"
          />
          <View style={styles.inputFooter}>
            <Text style={styles.charCount}>{inputText.length} characters</Text>
            {inputText.length > 0 && (
              <TouchableOpacity onPress={() => setInputText('')}>
                <Text style={styles.clearButton}>Clear</Text>
              </TouchableOpacity>
            )}
          </View>
        </View>

        {/* Translate Button */}
        <TouchableOpacity
          style={[styles.translateButton, isTranslating && styles.buttonDisabled]}
          onPress={handleTranslate}
          disabled={isTranslating || !inputText.trim()}
        >
          {isTranslating ? (
            <ActivityIndicator color="#FFFFFF" />
          ) : (
            <Text style={styles.translateButtonText}>Translate</Text>
          )}
        </TouchableOpacity>

        {/* Translation Result */}
        {translatedText && !error && (
          <View style={styles.resultSection}>
            <View style={styles.resultHeader}>
              <Text style={styles.sectionLabel}>Translation</Text>
              <TouchableOpacity onPress={clearCache}>
                <Text style={styles.clearButton}>Clear Cache</Text>
              </TouchableOpacity>
            </View>
            <View style={styles.resultContainer}>
              <Text style={styles.resultText} selectable>
                {translatedText}
              </Text>
            </View>
          </View>
        )}
      </ScrollView>

      {/* Language Picker Modals */}
      <LanguagePicker
        visible={showSourcePicker}
        languages={languages}
        selectedLanguage={sourceLang}
        excludeLanguage={targetLang}
        onSelect={(lang) => handleLanguageSelect(lang, true)}
        onClose={() => setShowSourcePicker(false)}
      />

      <LanguagePicker
        visible={showTargetPicker}
        languages={languages}
        selectedLanguage={targetLang}
        excludeLanguage={sourceLang}
        onSelect={(lang) => handleLanguageSelect(lang, false)}
        onClose={() => setShowTargetPicker(false)}
      />
    </KeyboardAvoidingView>
  );
}

// Language Picker Component
interface LanguagePickerProps {
  visible: boolean;
  languages: LanguageInfo[];
  selectedLanguage: string;
  excludeLanguage: string;
  onSelect: (language: string) => void;
  onClose: () => void;
}

function LanguagePicker({
  visible,
  languages,
  selectedLanguage,
  excludeLanguage,
  onSelect,
  onClose,
}: LanguagePickerProps) {
  const filteredLanguages = languages.filter((lang) => lang.code !== excludeLanguage);

  return (
    <Modal visible={visible} animationType="slide" transparent>
      <View style={styles.modalContainer}>
        <View style={styles.modalContent}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Select Language</Text>
            <TouchableOpacity onPress={onClose}>
              <Text style={styles.modalClose}>✕</Text>
            </TouchableOpacity>
          </View>

          <FlatList
            data={filteredLanguages}
            keyExtractor={(item) => item.code}
            renderItem={({ item }) => (
              <TouchableOpacity
                style={[
                  styles.languageItem,
                  item.code === selectedLanguage && styles.selectedLanguage,
                ]}
                onPress={() => onSelect(item.code)}
              >
                <View>
                  <Text style={[
                    styles.languageItemName,
                    item.isRTL && styles.rtlText
                  ]}>
                    {item.name}
                  </Text>
                  <Text style={[
                    styles.languageItemNative,
                    item.isRTL && styles.rtlText
                  ]}>
                    {item.nativeName}
                  </Text>
                </View>
                {item.code === selectedLanguage && (
                  <Text style={styles.checkmark}>✓</Text>
                )}
              </TouchableOpacity>
            )}
          />
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F5F5F5',
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#666666',
  },
  scrollContent: {
    padding: 16,
  },
  languageSelector: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
  },
  languageButton: {
    flex: 1,
    backgroundColor: '#FFFFFF',
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  languageLabel: {
    fontSize: 12,
    color: '#666666',
    marginBottom: 4,
  },
  languageName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333333',
  },
  swapButton: {
    marginHorizontal: 12,
    padding: 8,
  },
  swapIcon: {
    fontSize: 24,
    color: '#007AFF',
  },
  progressContainer: {
    marginBottom: 16,
  },
  progressItem: {
    marginBottom: 8,
  },
  progressText: {
    fontSize: 12,
    color: '#666666',
    marginBottom: 4,
  },
  progressBar: {
    height: 4,
    backgroundColor: '#E0E0E0',
    borderRadius: 2,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#007AFF',
  },
  inputSection: {
    marginBottom: 20,
  },
  sectionLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333333',
    marginBottom: 8,
  },
  input: {
    backgroundColor: '#FFFFFF',
    borderRadius: 8,
    padding: 12,
    minHeight: 120,
    fontSize: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  inputFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 4,
  },
  charCount: {
    fontSize: 12,
    color: '#666666',
  },
  clearButton: {
    fontSize: 14,
    color: '#007AFF',
  },
  translateButton: {
    backgroundColor: '#007AFF',
    borderRadius: 8,
    padding: 16,
    alignItems: 'center',
    marginBottom: 20,
  },
  buttonDisabled: {
    opacity: 0.6,
  },
  translateButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
  resultSection: {
    marginTop: 20,
  },
  resultHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  resultContainer: {
    backgroundColor: '#FFFFFF',
    borderRadius: 8,
    padding: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  resultText: {
    fontSize: 16,
    color: '#333333',
    lineHeight: 24,
  },
  modalContainer: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: '#FFFFFF',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    maxHeight: '80%',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#EEEEEE',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333333',
  },
  modalClose: {
    fontSize: 24,
    color: '#666666',
    padding: 4,
  },
  languageItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#EEEEEE',
  },
  selectedLanguage: {
    backgroundColor: '#F0F8FF',
  },
  languageItemName: {
    fontSize: 16,
    color: '#333333',
    marginBottom: 2,
  },
  languageItemNative: {
    fontSize: 14,
    color: '#666666',
  },
  rtlText: {
    textAlign: 'right',
  },
  checkmark: {
    fontSize: 20,
    color: '#007AFF',
    fontWeight: '600',
  },
});