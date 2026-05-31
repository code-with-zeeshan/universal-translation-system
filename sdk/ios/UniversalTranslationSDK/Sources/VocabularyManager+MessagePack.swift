// ios/UniversalTranslationSDK/Sources/VocabularyManager+MessagePack.swift

import Foundation
import MessagePack
import SWCompression
import OSLog

private let logger = Logger(subsystem: "com.universaltranslation.sdk", category: "VocabularyManager")

// MARK: - Updated VocabularyManager

extension VocabularyManager {
    
    // Updated load method with MessagePack support
    public func loadVocabulary(from path: String) async throws -> VocabularyPack {
        let url = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: url)
        
        // Determine format by extension or magic bytes
        if path.hasSuffix(".msgpack") || data.starts(with: [0xdc, 0x00]) || data.starts(with: [0xde]) {
            return try loadMessagePackVocabulary(from: data)
        } else if path.hasSuffix(".json") {
            return try loadJSONVocabulary(from: data)
        } else {
            // Try to detect format
            if let _ = try? loadMessagePackVocabulary(from: data) {
                return try loadMessagePackVocabulary(from: data)
            } else {
                return try loadJSONVocabulary(from: data)
            }
        }
    }
    
    private func loadMessagePackVocabulary(from data: Data) throws -> VocabularyPack {
        logger.info("Loading MessagePack vocabulary...")
        
        do {
            let decoder = MessagePackDecoder()
            let vocab = try decoder.decode(VocabularyPack.self, from: data)
            
            logger.info("Loaded vocabulary: \(vocab.name) v\(vocab.version)")
            logger.info("  - Tokens: \(vocab.tokens.count)")
            logger.info("  - Subwords: \(vocab.subwords.count)")
            logger.info("  - Languages: \(vocab.languages.joined(separator: ", "))")
            
            return vocab
        } catch {
            logger.error("Failed to decode MessagePack vocabulary: \(error)")
            throw TranslationError.vocabularyNotLoaded
        }
    }
    
    private func loadJSONVocabulary(from data: Data) throws -> VocabularyPack {
        logger.info("Loading JSON vocabulary...")
        
        do {
            let decoder = JSONDecoder()
            let vocab = try decoder.decode(VocabularyPack.self, from: data)
            
            logger.info("Loaded vocabulary: \(vocab.name) v\(vocab.version)")
            
            return vocab
        } catch {
            logger.error("Failed to decode JSON vocabulary: \(error)")
            throw TranslationError.vocabularyNotLoaded
        }
    }
    
    // Download and process vocabulary with compression support
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
            
            // Check if data is compressed
            let processedData: Data
            if isCompressed(data) {
                processedData = try decompressData(data)
                logger.info("Decompressed vocabulary data")
            } else {
                processedData = data
            }
            
            // Create vocabulary directory if needed
            let vocabDir = URL(fileURLWithPath: pack.localPath).deletingLastPathComponent()
            try FileManager.default.createDirectory(at: vocabDir, withIntermediateDirectories: true)
            
            // Save to file
            let fileURL = URL(fileURLWithPath: pack.localPath)
            try processedData.write(to: fileURL)
            
            // Verify the download
            let savedPack = try await loadVocabulary(from: pack.localPath)
            logger.info("Downloaded and verified vocabulary pack: \(savedPack.name)")
            
        } catch {
            logger.error("Failed to download vocabulary: \(error)")
            throw TranslationError.networkError(error)
        }
    }
    
    private func isCompressed(_ data: Data) -> Bool {
        // Check for common compression headers
        if data.count < 4 { return false }
        
        let header = data.prefix(4)
        
        // LZ4 frame magic number
        if header.starts(with: [0x04, 0x22, 0x4D, 0x18]) { return true }
        
        // Gzip magic number
        if header.starts(with: [0x1F, 0x8B]) { return true }
        
        // Zlib header
        if header[0] == 0x78 && (header[1] == 0x01 || header[1] == 0x9C || header[1] == 0xDA) {
            return true
        }
        
        return false
    }
    
    private func decompressData(_ data: Data) throws -> Data {
        // Try different decompression methods
        
        // LZ4
        if data.starts(with: [0x04, 0x22, 0x4D, 0x18]) {
            return try LZ4.decompress(data: data)
        }
        
        // Gzip
        if let decompressed = try? GzipArchive.unarchive(archive: data) {
            return decompressed
        }
        
        // Zlib
        if let decompressed = try? ZlibArchive.unarchive(archive: data) {
            return decompressed
        }
        
        throw TranslationError.compressionFailed
    }
    
    // Create edge-optimized vocabulary pack
    public func createEdgePack(from fullPack: VocabularyPack) -> EdgeVocabularyPack {
        return EdgeVocabularyPack(from: fullPack)
    }
}

// MARK: - Vocabulary Cache Manager

public actor VocabularyCacheManager {
    private var cache: [String: EdgeVocabularyPack] = [:]
    private var cacheOrder: [String] = []
    private let maxCacheSize: Int
    private let logger = Logger(subsystem: "com.universaltranslation.sdk", category: "VocabCache")
    
    public init(maxCacheSize: Int = 3) {
        self.maxCacheSize = maxCacheSize
    }
    
    public func get(_ key: String) -> EdgeVocabularyPack? {
        if let pack = cache[key] {
            // Move to end (LRU)
            cacheOrder.removeAll { $0 == key }
            cacheOrder.append(key)
            return pack
        }
        return nil
    }
    
    public func set(_ key: String, pack: EdgeVocabularyPack) {
        // Check if we need to evict
        if cache.count >= maxCacheSize && cache[key] == nil {
            if let oldestKey = cacheOrder.first {
                cache.removeValue(forKey: oldestKey)
                cacheOrder.removeFirst()
                logger.info("Evicted vocabulary pack: \(oldestKey)")
            }
        }
        
        cache[key] = pack
        cacheOrder.removeAll { $0 == key }
        cacheOrder.append(key)
        logger.info("Cached vocabulary pack: \(key)")
    }
    
    public func clear() {
        cache.removeAll()
        cacheOrder.removeAll()
        logger.info("Cleared vocabulary cache")
    }
    
    public var currentCacheSize: Int {
        cache.count
    }
}