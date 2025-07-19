// ios/UniversalTranslationSDK/Sources/Models.swift

import Foundation

// MARK: - Vocabulary Pack

public struct VocabularyPack: Codable {
    public let name: String
    public let languages: [String]
    public let downloadURL: String
    public let localPath: String
    public let sizeMB: Double
    public let version: String
    public let tokens: [String: Int]
    public let subwords: [String: Int]
    public let specialTokens: [String: Int]
    
    public var needsDownload: Bool {
        return !FileManager.default.fileExists(atPath: localPath)
    }
    
    public var url: URL? {
        return URL(string: downloadURL)
    }
}

// MARK: - Translation Response

public struct TranslationResponse: Codable {
    public let translation: String
    public let targetLang: String
    public let confidence: Double?
    public let alternativeTranslations: [String]?
    
    private enum CodingKeys: String, CodingKey {
        case translation
        case targetLang = "target_lang"
        case confidence
        case alternativeTranslations = "alternative_translations"
    }
}

// MARK: - Translation Request

public struct TranslationRequest {
    public let text: String
    public let sourceLang: String
    public let targetLang: String
    public let options: TranslationOptions?
    
    public init(text: String, from sourceLang: String, to targetLang: String, options: TranslationOptions? = nil) {
        self.text = text
        self.sourceLang = sourceLang
        self.targetLang = targetLang
        self.options = options
    }
}

// MARK: - Translation Options

public struct TranslationOptions {
    public let formality: Formality?
    public let domain: TranslationDomain?
    public let preserveFormatting: Bool
    
    public enum Formality: String {
        case formal
        case informal
        case auto
    }
    
    public enum TranslationDomain: String {
        case general
        case medical
        case legal
        case technical
        case business
    }
    
    public init(formality: Formality? = nil, domain: TranslationDomain? = nil, preserveFormatting: Bool = true) {
        self.formality = formality
        self.domain = domain
        self.preserveFormatting = preserveFormatting
    }
}

// MARK: - Language Info

public struct LanguageInfo {
    public let code: String
    public let name: String
    public let nativeName: String
    public let script: String
    public let isRTL: Bool
    
    public static let supportedLanguages: [LanguageInfo] = [
        LanguageInfo(code: "en", name: "English", nativeName: "English", script: "Latin", isRTL: false),
        LanguageInfo(code: "es", name: "Spanish", nativeName: "Español", script: "Latin", isRTL: false),
        LanguageInfo(code: "fr", name: "French", nativeName: "Français", script: "Latin", isRTL: false),
        LanguageInfo(code: "de", name: "German", nativeName: "Deutsch", script: "Latin", isRTL: false),
        LanguageInfo(code: "zh", name: "Chinese", nativeName: "中文", script: "Han", isRTL: false),
        LanguageInfo(code: "ja", name: "Japanese", nativeName: "日本語", script: "Kana/Kanji", isRTL: false),
        LanguageInfo(code: "ko", name: "Korean", nativeName: "한국어", script: "Hangul", isRTL: false),
        LanguageInfo(code: "ar", name: "Arabic", nativeName: "العربية", script: "Arabic", isRTL: true),
        LanguageInfo(code: "hi", name: "Hindi", nativeName: "हिन्दी", script: "Devanagari", isRTL: false),
        LanguageInfo(code: "ru", name: "Russian", nativeName: "Русский", script: "Cyrillic", isRTL: false),
        LanguageInfo(code: "pt", name: "Portuguese", nativeName: "Português", script: "Latin", isRTL: false),
    ]
}