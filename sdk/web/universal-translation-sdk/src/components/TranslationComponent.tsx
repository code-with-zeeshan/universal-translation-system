// src/components/TranslationComponent.tsx
import React, { useState, useEffect, useCallback } from 'react';
import { TranslationClient } from '../index';

export interface TranslationComponentProps {
  defaultSourceLang?: string;
  defaultTargetLang?: string;
  decoderUrl?: string;
  className?: string;
  onTranslation?: (result: { text: string; translation: string }) => void;
}

export const TranslationComponent: React.FC<TranslationComponentProps> = ({
  defaultSourceLang = 'en',
  defaultTargetLang = 'es',
  decoderUrl,
  className = '',
  onTranslation
}) => {
  const [client] = useState(() => new TranslationClient({ decoderUrl }));
  const [inputText, setInputText] = useState('');
  const [translatedText, setTranslatedText] = useState('');
  const [isTranslating, setIsTranslating] = useState(false);
  const [isInitializing, setIsInitializing] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sourceLang, setSourceLang] = useState(defaultSourceLang);
  const [targetLang, setTargetLang] = useState(defaultTargetLang);
  
  useEffect(() => {
    // Initialize client
    client.initialize()
      .then(() => setIsInitializing(false))
      .catch(err => {
        setError(`Failed to initialize: ${err.message}`);
        setIsInitializing(false);
      });
  }, [client]);
  
  const handleTranslate = useCallback(async () => {
    if (!inputText.trim() || isTranslating || isInitializing) return;
    
    setIsTranslating(true);
    setError(null);
    
    try {
      const result = await client.translate({
        text: inputText,
        sourceLang,
        targetLang,
      });
      
      setTranslatedText(result.translation);
      
      if (onTranslation) {
        onTranslation({
          text: inputText,
          translation: result.translation
        });
      }
    } catch (err: any) {
      console.error('Translation failed:', err);
      setError(err.message || 'Translation failed');
      setTranslatedText('');
    } finally {
      setIsTranslating(false);
    }
  }, [client, inputText, sourceLang, targetLang, onTranslation]);
  
  const handleSwapLanguages = useCallback(() => {
    setSourceLang(targetLang);
    setTargetLang(sourceLang);
    
    if (translatedText) {
      setInputText(translatedText);
      setTranslatedText('');
    }
  }, [sourceLang, targetLang, translatedText]);
  
  const supportedLanguages = client.getSupportedLanguages();
  
  if (isInitializing) {
    return (
      <div className={`translation-container ${className}`}>
        <div className="translation-loading">Initializing translation engine...</div>
      </div>
    );
  }
  
  return (
    <div className={`translation-container ${className}`}>
      <div className="translation-header">
        <div className="language-selector">
          <select 
            value={sourceLang} 
            onChange={(e) => setSourceLang(e.target.value)}
            disabled={isTranslating}
            className="language-select"
          >
            {supportedLanguages.map(lang => (
              <option key={lang} value={lang}>
                {lang.toUpperCase()}
              </option>
            ))}
          </select>
          
          <button 
            onClick={handleSwapLanguages}
            disabled={isTranslating}
            className="swap-button"
            aria-label="Swap languages"
          >
            ⇄
          </button>
          
          <select 
            value={targetLang} 
            onChange={(e) => setTargetLang(e.target.value)}
            disabled={isTranslating}
            className="language-select"
          >
            {supportedLanguages.filter(lang => lang !== sourceLang).map(lang => (
              <option key={lang} value={lang}>
                {lang.toUpperCase()}
              </option>
            ))}
          </select>
        </div>
      </div>
      
      <div className="translation-body">
        <div className="input-section">
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Enter text to translate"
            disabled={isTranslating}
            className="translation-input"
            rows={5}
          />
          <div className="input-footer">
            <span className="char-count">{inputText.length} characters</span>
          </div>
        </div>
        
        <button
          onClick={handleTranslate}
          disabled={isTranslating || !inputText.trim()}
          className="translate-button"
        >
          {isTranslating ? 'Translating...' : 'Translate'}
        </button>
        
        {error && (
          <div className="error-message">
            Error: {error}
          </div>
        )}
        
        {translatedText && !error && (
          <div className="result-section">
            <h3>Translation:</h3>
            <div className="translation-result">
              {translatedText}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default TranslationComponent;