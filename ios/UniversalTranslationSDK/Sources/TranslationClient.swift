// ios/UniversalTranslationSDK/Sources/TranslationClient.swift

import Foundation
import OSLog
import Network

private let logger = Logger(subsystem: "com.universaltranslation.sdk", category: "TranslationClient")

// MARK: - Translation Client

@available(iOS 15.0, macOS 12.0, *)
public actor TranslationClient {
    private let encoder: TranslationEncoder
    private let decoderURL: URL
    private let session: URLSession
    private let monitor: NWPathMonitor
    
    // Cache for translations
    private var translationCache: [String: String] = [:]
    private let maxCacheSize = 100
    
    public init(decoderURL: String = "https://api.yourdomain.com/decode") throws {
        guard let url = URL(string: decoderURL) else {
            throw TranslationError.networkError(URLError(.badURL))
        }
        
        self.encoder = try TranslationEncoder()
        self.decoderURL = url
        
        // Configure session
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.httpAdditionalHeaders = [
            "User-Agent": "UniversalTranslationSDK/1.0"
        ]
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        
        self.session = URLSession(configuration: config)
        
        // Setup network monitoring
        self.monitor = NWPathMonitor()
        self.monitor.start(queue: .global(qos: .background))
        
        logger.info("TranslationClient initialized with decoder URL: \(decoderURL)")
    }
    
    deinit {
        monitor.cancel()
    }
    
    // MARK: - Public Methods
    
    public func translate(
        text: String,
        from sourceLang: String,
        to targetLang: String,
        options: TranslationOptions? = nil
    ) async throws -> TranslationResponse {
        
        // Check network connectivity
        guard monitor.currentPath.status == .satisfied else {
            throw TranslationError.networkError(URLError(.notConnectedToInternet))
        }
        
        // Check cache
        let cacheKey = "\(sourceLang)-\(targetLang):\(text)"
        if let cachedTranslation = translationCache[cacheKey] {
            logger.info("Returning cached translation")
            return TranslationResponse(
                translation: cachedTranslation,
                targetLang: targetLang,
                confidence: 1.0,
                alternativeTranslations: nil
            )
        }
        
        // Encode locally
        logger.info("Encoding text: \(text.prefix(50))...")
        let encodedData = try await encoder.encode(
            text: text,
            sourceLang: sourceLang,
            targetLang: targetLang
        )
        
        // Send to decoder
        let response = try await sendToDecoder(
            encodedData: encodedData,
            targetLang: targetLang,
            options: options
        )
        
        // Cache the result
        addToCache(key: cacheKey, translation: response.translation)
        
        return response
    }
    
    public func translateBatch(
        texts: [String],
        from sourceLang: String,
        to targetLang: String,
        options: TranslationOptions? = nil
    ) async throws -> [TranslationResponse] {
        
        // Process in parallel with limited concurrency
        return try await withThrowingTaskGroup(of: (Int, TranslationResponse).self) { group in
            for (index, text) in texts.enumerated() {
                group.addTask { [self] in
                    let response = try await self.translate(
                        text: text,
                        from: sourceLang,
                        to: targetLang,
                        options: options
                    )
                    return (index, response)
                }
            }
            
            // Collect results in order
            var results = Array<TranslationResponse?>(repeating: nil, count: texts.count)
            for try await (index, response) in group {
                results[index] = response
            }
            
            return results.compactMap { $0 }
        }
    }
    
    public func getSupportedLanguages() -> [LanguageInfo] {
        return LanguageInfo.supportedLanguages
    }
    
    public func isLanguageSupported(_ languageCode: String) -> Bool {
        return VocabularyManager.supportedLanguages.contains(languageCode)
    }
    
    public func clearCache() {
        translationCache.removeAll()
        logger.info("Translation cache cleared")
    }
    
    public func getMemoryUsage() async -> Int {
        await encoder.getMemoryUsage()
    }
    
    // MARK: - Private Methods
    
    private func sendToDecoder(
        encodedData: Data,
        targetLang: String,
        options: TranslationOptions?
    ) async throws -> TranslationResponse {
        
        var request = URLRequest(url: decoderURL)
        request.httpMethod = "POST"
        request.httpBody = encodedData
        request.setValue(targetLang, forHTTPHeaderField: "X-Target-Language")
        request.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
        
        // Add options headers if provided
        if let formality = options?.formality {
            request.setValue(formality.rawValue, forHTTPHeaderField: "X-Formality")
        }
        if let domain = options?.domain {
            request.setValue(domain.rawValue, forHTTPHeaderField: "X-Domain")
        }
        
        logger.info("Sending request to decoder (\(encodedData.count) bytes)")
        
        do {
            let (data, response) = try await session.data(for: request)
            
            guard let httpResponse = response as? HTTPURLResponse else {
                throw TranslationError.networkError(URLError(.badServerResponse))
            }
            
            switch httpResponse.statusCode {
            case 200:
                let decoder = JSONDecoder()
                let result = try decoder.decode(TranslationResponse.self, from: data)
                logger.info("Translation successful")
                return result
                
            case 429:
                throw TranslationError.networkError(URLError(.resourceUnavailable))
                
            case 400...499:
                throw TranslationError.decodingFailed
                
            case 500...599:
                throw TranslationError.networkError(URLError(.cannotConnectToHost))
                
            default:
                throw TranslationError.networkError(URLError(.unknown))
            }
            
        } catch {
            logger.error("Decoder request failed: \(error)")
            throw TranslationError.networkError(error)
        }
    }
    
    private func addToCache(key: String, translation: String) {
        // Implement LRU cache
        if translationCache.count >= maxCacheSize {
            // Remove oldest entry (simplified - in production use proper LRU)
            if let firstKey = translationCache.keys.first {
                translationCache.removeValue(forKey: firstKey)
            }
        }
        
        translationCache[key] = translation
    }
}