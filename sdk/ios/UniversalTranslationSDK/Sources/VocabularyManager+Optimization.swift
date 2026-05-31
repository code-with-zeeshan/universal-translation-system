// ios/UniversalTranslationSDK/Sources/VocabularyManager+Optimization.swift

import Foundation
import OSLog

private let logger = Logger(subsystem: "com.universaltranslation.sdk", category: "VocabularyManager")

extension VocabularyManager {
    // Prefetch vocabularies based on user preferences
    public func prefetchVocabulariesForUserLanguages() async {
        let preferredLanguages = Locale.preferredLanguages
            .compactMap { Locale(identifier: $0).languageCode }
            .prefix(3)
        
        logger.info("Prefetching vocabularies for user languages: \(preferredLanguages)")
        
        await withTaskGroup(of: Void.self) { group in
            for lang in preferredLanguages {
                group.addTask { [weak self] in
                    await self?.prefetchVocabulary(for: lang)
                }
            }
        }
    }
    
    private func prefetchVocabulary(for language: String) async {
        do {
            // Get vocabulary pack for common pairs
            let commonPairs = ["en", language] // Most users translate to/from English
            
            for sourceLang in commonPairs {
                for targetLang in commonPairs where sourceLang != targetLang {
                    let pack = try await getVocabularyForPair(
                        source: sourceLang,
                        target: targetLang
                    )
                    
                    if pack.needsDownload {
                        logger.info("Prefetching vocabulary: \(pack.name)")
                        try await downloadVocabulary(pack)
                    }
                }
            }
        } catch {
            logger.warning("Failed to prefetch vocabulary for \(language): \(error)")
        }
    }
    
    // Memory-efficient loading for large vocabularies
    public func loadVocabularyEfficient(from path: String) async throws -> VocabularyPack {
        let url = URL(fileURLWithPath: path)
        let fileSize = try FileManager.default.attributesOfItem(atPath: path)[.size] as? Int ?? 0
        
        if fileSize > 5 * 1024 * 1024 { // 5MB
            logger.info("Using memory-mapped loading for large vocabulary: \(fileSize / 1024 / 1024)MB")
            return try await loadVocabularyMemoryMapped(from: url)
        } else {
            return try await loadVocabulary(from: path)
        }
    }
    
    private func loadVocabularyMemoryMapped(from url: URL) async throws -> VocabularyPack {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        let decoder = JSONDecoder()
        return try decoder.decode(VocabularyPack.self, from: data)
    }
}

@available(iOS 15.0, macOS 12.0, *)
public class VocabularyManager {
    public static let supportedLanguages = ["en", "es", "fr", "de", "zh", "ja", "ko", "ar", "hi", "ru", "pt", "it", "tr", "th", "vi", "pl", "uk", "nl", "id", "sv"]
    
    private static let languageToPack: [String: String] = [
        "en": "latin", "es": "latin", "fr": "latin", "de": "latin",
        "it": "latin", "pt": "latin", "nl": "latin", "sv": "latin",
        "zh": "cjk", "ja": "cjk", "ko": "cjk",
        "ar": "arabic", "hi": "devanagari",
        "ru": "cyrillic", "uk": "cyrillic",
        "th": "thai", "vi": "latin", "pl": "latin",
        "tr": "latin", "id": "latin"
    ]
    
    private let vocabDirectory: URL
    private let session: URLSession
    
    public init() {
        // Setup vocabulary directory
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        self.vocabDirectory = documentsPath.appendingPathComponent("Vocabularies")
        
        // Create directory if needed
        try? FileManager.default.createDirectory(at: vocabDirectory, withIntermediateDirectories: true)
        
        // Configure URL session
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 300
        config.allowsCellularAccess = true
        config.waitsForConnectivity = true
        
        self.session = URLSession(configuration: config)
        
        logger.info("VocabularyManager initialized with directory: \(self.vocabDirectory.path)")
    }
    
    public func getVocabularyForPair(source: String, target: String) async throws -> VocabularyPack {
        // Determine pack name
        let sourcePack = Self.languageToPack[source] ?? "latin"
        let targetPack = Self.languageToPack[target] ?? "latin"
        let packName = targetPack != "latin" ? targetPack : sourcePack
        
        // Create vocabulary pack info
        let localPath = vocabDirectory.appendingPathComponent("\(packName).vocab").path
        
        let pack = VocabularyPack(
            name: packName,
            languages: getLanguagesForPack(packName),
            downloadURL: getDownloadURL(for: packName),
            localPath: localPath,
            sizeMB: getPackSize(packName),
            version: "1.0",
            tokens: [:],
            subwords: [:],
            specialTokens: [:]
        )
        
        logger.info("Selected vocabulary pack: \(packName) for \(source)->\(target)")
        
        return pack
    }
    
    public func downloadVocabulary(_ pack: VocabularyPack) async throws {
        guard let url = pack.url else {
            throw TranslationError.networkError(URLError(.badURL))
        }
        
        logger.info("Downloading vocabulary pack: \(pack.name) from \(url)")
        
        do {
            let (data, response) = try await session.data(from: url)
            
            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                throw TranslationError.networkError(URLError(.badServerResponse))
            }
            
            // Save to file
            let tempURL = URL(fileURLWithPath: pack.localPath + ".tmp")
            try data.write(to: tempURL)
            
            // Atomic move
            _ = try FileManager.default.replaceItemAt(URL(fileURLWithPath: pack.localPath), withItemAt: tempURL)
            
            logger.info("Downloaded vocabulary pack: \(pack.name) (\(data.count) bytes)")
            
        } catch {
            logger.error("Failed to download vocabulary: \(error)")
            throw TranslationError.networkError(error)
        }
    }
    
    public func loadVocabulary(from path: String) async throws -> VocabularyPack {
        let url = URL(fileURLWithPath: path)
        
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        let vocab = try decoder.decode(VocabularyPack.self, from: data)
        
        logger.info("Loaded vocabulary with \(vocab.tokens.count) tokens")
        
        return vocab
    }
    
    public func deleteVocabulary(packName: String) throws {
        let path = vocabDirectory.appendingPathComponent("\(packName).vocab")
        try FileManager.default.removeItem(at: path)
        logger.info("Deleted vocabulary pack: \(packName)")
    }
    
    public func getDownloadedPacks() -> [String] {
        do {
            let files = try FileManager.default.contentsOfDirectory(at: vocabDirectory, includingPropertiesForKeys: nil)
            return files.compactMap { url in
                url.pathExtension == "vocab" ? url.deletingPathExtension().lastPathComponent : nil
            }
        } catch {
            logger.error("Failed to list vocabulary packs: \(error)")
            return []
        }
    }
    
    // MARK: - Private Helpers
    
    private func getLanguagesForPack(_ packName: String) -> [String] {
        switch packName {
        case "latin":
            return ["en", "es", "fr", "de", "it", "pt", "nl", "sv", "pl", "vi", "tr", "id"]
        case "cjk":
            return ["zh", "ja", "ko"]
        case "arabic":
            return ["ar"]
        case "devanagari":
            return ["hi"]
        case "cyrillic":
            return ["ru", "uk"]
        case "thai":
            return ["th"]
        default:
            return []
        }
    }
    
    private func getDownloadURL(for packName: String) -> String {
        // Replace with your actual CDN URL
        return "https://cdn.yourdomain.com/vocabs/\(packName)_v1.0.vocab"
    }
    
    private func getPackSize(_ packName: String) -> Double {
        switch packName {
        case "latin": return 5.0
        case "cjk": return 8.0
        case "arabic": return 3.0
        case "devanagari": return 3.0
        case "cyrillic": return 4.0
        case "thai": return 2.0
        default: return 5.0
        }
    }
}