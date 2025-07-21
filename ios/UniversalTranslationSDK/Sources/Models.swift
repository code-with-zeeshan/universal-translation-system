// ios/UniversalTranslationSDK/Sources/Models.swift

import Foundation
import MessagePack

// MARK: - Complete Vocabulary Pack 

public struct VocabularyPack: Codable {
    public let name: String
    public let version: String
    public let languages: [String]
    public let tokens: [String: Int]
    public let embeddings: [String: [Float]]?  //  Critical for quality!
    public let compression: String?  //  "int8", "fp16", etc.
    public let subwords: [String: Int]
    public let specialTokens: [String: Int]
    public let metadata: VocabularyMetadata 
    
    // Computed properties
    public var totalTokens: Int {
        tokens.count + subwords.count + specialTokens.count
    }
    
    public var sizeMB: Double {
        metadata.sizeMB
    }
    
    public var needsDownload: Bool {
        !FileManager.default.fileExists(atPath: localPath)
    }
    
    // Local storage path
    public var localPath: String {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return documentsPath.appendingPathComponent("Vocabularies/\(name)_v\(version).msgpack").path
    }
    
    // Download URL
    public var downloadURL: String {
        let baseURL = ProcessInfo.processInfo.environment["VOCAB_CDN_URL"] 
            ?? "https://cdn.universaltranslation.com/vocabs"
        return "\(baseURL)/\(name)_v\(version).msgpack"
    }
    
    public var url: URL? {
        URL(string: downloadURL)
    }
}

public struct VocabularyMetadata: Codable {
    public let totalTokens: Int
    public let sizeMB: Double
    public let coveragePercentage: Double
    public let compressionRatio: Double
    public let oovRate: Double
    public let vocabSize: Int
    public let modelType: String
    public let characterCoverage: Double
    
    private enum CodingKeys: String, CodingKey {
        case totalTokens = "total_tokens"
        case sizeMB = "size_mb"
        case coveragePercentage = "coverage_percentage"
        case compressionRatio = "compression_ratio"
        case oovRate = "oov_rate"
        case vocabSize = "vocab_size"
        case modelType = "model_type"
        case characterCoverage = "character_coverage"
    }
}

// MARK: - Edge Vocabulary Pack 

public struct EdgeVocabularyPack {
    public let name: String
    public let tokens: [String: Int]
    public let specialTokens: [String: Int]
    public let subwords: [String: Int]
    private let prefixTree: PrefixTree
    
    public init(from fullPack: VocabularyPack, maxTokens: Int = 8000) {
        self.name = fullPack.name
        self.specialTokens = fullPack.specialTokens
        
        // Keep only most essential tokens
        let sortedTokens = fullPack.tokens.sorted { $0.value < $1.value }
        self.tokens = Dictionary(uniqueKeysWithValues: sortedTokens.prefix(maxTokens))
        
        // Keep essential subwords
        let essentialSubwords = fullPack.subwords.filter { $0.value < maxTokens * 2 }
        self.subwords = essentialSubwords
        
        // Build prefix tree
        self.prefixTree = PrefixTree()
        for (token, id) in tokens {
            prefixTree.insert(word: token, id: id)
        }
        for (subword, id) in subwords {
            prefixTree.insert(word: subword, id: id)
        }
    }
}

// MARK: - Prefix Tree 

class PrefixTree {
    class Node {
        var children: [Character: Node] = [:]
        var tokenId: Int?
    }
    
    private let root = Node()
    
    func insert(word: String, id: Int) {
        var current = root
        for char in word {
            if current.children[char] == nil {
                current.children[char] = Node()
            }
            current = current.children[char]!
        }
        current.tokenId = id
    }
    
    func findLongestMatch(in word: String, startingAt start: Int, withPrefix prefix: String) -> (length: Int, tokenId: Int) {
        var current = root
        var longestMatch = (length: 0, tokenId: 0)
        
        // First, traverse the prefix
        for char in prefix {
            guard let next = current.children[char] else {
                return (0, 0)
            }
            current = next
        }
        
        // Then find the longest match in the word
        let chars = Array(word)
        for i in start..<chars.count {
            guard let next = current.children[chars[i]] else {
                break
            }
            current = next
            
            if let tokenId = current.tokenId {
                longestMatch = (length: i - start + 1, tokenId: tokenId)
            }
        }
        
        return longestMatch
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
        LanguageInfo(code: "it", name: "Italian", nativeName: "Italiano", script: "Latin", isRTL: false),
        LanguageInfo(code: "tr", name: "Turkish", nativeName: "Türkçe", script: "Latin", isRTL: false),
        LanguageInfo(code: "th", name: "Thai", nativeName: "ไทย", script: "Thai", isRTL: false),
        LanguageInfo(code: "vi", name: "Vietnamese", nativeName: "Tiếng Việt", script: "Latin", isRTL: false),
        LanguageInfo(code: "pl", name: "Polish", nativeName: "Polski", script: "Latin", isRTL: false),
        LanguageInfo(code: "uk", name: "Ukrainian", nativeName: "Українська", script: "Cyrillic", isRTL: false),
        LanguageInfo(code: "nl", name: "Dutch", nativeName: "Nederlands", script: "Latin", isRTL: false),
        LanguageInfo(code: "id", name: "Indonesian", nativeName: "Bahasa Indonesia", script: "Latin", isRTL: false),
        LanguageInfo(code: "sv", name: "Swedish", nativeName: "Svenska", script: "Latin", isRTL: false),
    ]
}

// Make LanguageInfo identifiable for SwiftUI
extension LanguageInfo: Identifiable {
    public var id: String { code }
}