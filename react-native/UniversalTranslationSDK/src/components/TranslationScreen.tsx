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
}

export function TranslationScreen({ decoderUrl }: TranslationScreenProps) {
  const {
    translate,
    isTranslating,
    error,
    getSupportedLanguages,
    clearCache,
  } = useTranslation({ decoderUrl });

  const [inputText, setInputText] = useState('');
  const [translatedText, setTranslatedText] = useState('');
  const [sourceLang, setSourceLang] = useState('en');
  const [targetLang, setTargetLang] = useState('es');
  const [languages, setLanguages] = useState<LanguageInfo[]>([]);
  const [showSourcePicker, setShowSourcePicker] = useState(false);
  const [showTargetPicker, setShowTargetPicker] = useState(false);

  useEffect(() => {
    loadLanguages();
  }, []);

  const loadLanguages = async () => {
    try {
      const langs = await getSupportedLanguages();
      setLanguages(langs);
    } catch (err) {
      console.error('Failed to load languages:', err);
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
      Alert.alert('Translation Error', err.message);
    }
  };

  const swapLanguages = () => {
    setSourceLang(targetLang);
    setTargetLang(sourceLang);
    if (translatedText) {
      setInputText(translatedText);
      setTranslatedText('');
    }
  };

  const getLanguageName = (code: string) => {
    return languages.find((lang) => lang.code === code)?.name || code;
  };

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
          <Text style={styles.charCount}>{inputText.length} characters</Text>
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

        {/* Error Display */}
        {error && (
          <View style={styles.errorContainer}>
            <Text style={styles.errorText}>{error}</Text>
          </View>
        )}

        {/* Translation Result */}
        {translatedText && !error && (
          <View style={styles.resultSection}>
            <View style={styles.resultHeader}>
              <Text style={styles.sectionLabel}>Translation</Text>
              <TouchableOpacity onPress={() => clearCache()}>
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
        onSelect={(lang) => {
          setSourceLang(lang);
          setShowSourcePicker(false);
        }}
        onClose={() => setShowSourcePicker(false)}
      />

      <LanguagePicker
        visible={showTargetPicker}
        languages={languages}
        selectedLanguage={targetLang}
        excludeLanguage={sourceLang}
        onSelect={(lang) => {
          setTargetLang(lang);
          setShowTargetPicker(false);
        }}
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
                  <Text style={styles.languageItemName}>{item.name}</Text>
                  <Text style={styles.languageItemNative}>{item.nativeName}</Text>
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
  charCount: {
    fontSize: 12,
    color: '#666666',
    marginTop: 4,
    textAlign: 'right',
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
  errorContainer: {
    backgroundColor: '#FFE5E5',
    borderRadius: 8,
    padding: 12,
    marginBottom: 20,
  },
  errorText: {
    color: '#CC0000',
    fontSize: 14,
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
  clearButton: {
    fontSize: 14,
    color: '#007AFF',
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
  checkmark: {
    fontSize: 20,
    color: '#007AFF',
    fontWeight: '600',
  },
});