// ios/UniversalTranslationSDK/Sources/TranslationClient+Background.swift

import Foundation
import OSLog
import Network
import BackgroundTasks

private let logger = Logger(subsystem: "com.universaltranslation.sdk", category: "TranslationClient")

// MARK: - Coordinator Status Models

struct CoordinatorStatusResponse: Codable {
    let singleDecoder: Bool
    let decoderPoolSize: Int
    let healthyDecoders: Int
    let decoders: [CoordinatorDecoderInfo]
    
    enum CodingKeys: String, CodingKey {
        case singleDecoder = "single_decoder"
        case decoderPoolSize = "decoder_pool_size"
        case healthyDecoders = "healthy_decoders"
        case decoders
    }
}

struct CoordinatorDecoderInfo: Codable {
    let nodeId: String
    let endpoint: String
    
    enum CodingKeys: String, CodingKey {
        case nodeId = "node_id"
        case endpoint
    }
}

@available(iOS 15.0, *)
extension TranslationClient {
    
    static let backgroundTaskIdentifier = "com.universal.translation.process"
    
    public func enableBackgroundTranslation() {
        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: Self.backgroundTaskIdentifier,
            using: nil
        ) { task in
            self.handleBackgroundTranslation(task: task as! BGProcessingTask)
        }
        
        scheduleBackgroundTranslation()
    }
    
    private func scheduleBackgroundTranslation() {
        let request = BGProcessingTaskRequest(identifier: Self.backgroundTaskIdentifier)
        request.requiresNetworkConnectivity = true
        request.requiresExternalPower = false
        request.earliestBeginDate = Date(timeIntervalSinceNow: 15 * 60)
        
        do {
            try BGTaskScheduler.shared.submit(request)
            logger.info("Background translation task scheduled")
        } catch {
            logger.error("Failed to schedule background task: \(error)")
        }
    }
    
    private func handleBackgroundTranslation(task: BGProcessingTask) {
        scheduleBackgroundTranslation()
        
        task.expirationHandler = {
            logger.info("Background task expired")
            task.setTaskCompleted(success: false)
        }
        
        Task {
            do {
                let success = await processPendingTranslations()
                await vocabularyManager.prefetchVocabulariesForUserLanguages()
                task.setTaskCompleted(success: success)
            } catch {
                logger.error("Background task failed: \(error)")
                task.setTaskCompleted(success: false)
            }
        }
    }
    
    private func processPendingTranslations() async -> Bool {
        logger.info("Processing pending translations in background")
        return true
    }
}

// MARK: - Translation Client

@available(iOS 15.0, macOS 12.0, *)
public actor TranslationClient {
    private let encoder: TranslationEncoder
    private let decoderURL: URL
    private let coordinatorURL: URL?
    private var localDecoderURL: URL?
    private let preferLocal: Bool
    private let hfRepo: String
    private var effectiveDecoderURL: URL
    private var useCoordinator: Bool = false
    private var localDecoderAvailable: Bool = false
    private let session: URLSession
    private let monitor: NWPathMonitor
    
    private var translationCache: [String: String] = [:]
    private let maxCacheSize = 100
    
    public var vocabularyManager: VocabularyManager
    
    public init(
        decoderURL: String = "https://api.yourdomain.com/decode",
        coordinatorURL: String? = nil,
        localDecoderURL: String? = nil,
        preferLocal: Bool = true,
        hfRepo: String = "your-org/universal-translation-system"
    ) throws {
        guard let url = URL(string: decoderURL) else {
            throw TranslationError.networkError(URLError(.badURL))
        }
        
        self.encoder = try TranslationEncoder()
        self.vocabularyManager = VocabularyManager()
        self.decoderURL = url
        self.effectiveDecoderURL = url
        self.coordinatorURL = coordinatorURL.flatMap { URL(string: $0) }
        self.localDecoderURL = localDecoderURL.flatMap { URL(string: $0) }
        self.preferLocal = preferLocal
        self.hfRepo = hfRepo
        
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.httpAdditionalHeaders = ["User-Agent": "UniversalTranslationSDK/1.0"]
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        
        self.session = URLSession(configuration: config)
        
        self.monitor = NWPathMonitor()
        self.monitor.start(queue: .global(qos: .background))
        
        logger.info("TranslationClient initialized with decoder URL: \(decoderURL)")
        
        if self.localDecoderURL != nil && self.preferLocal {
            Task { await self.checkLocalDecoder() }
        }
        if self.coordinatorURL != nil {
            Task { await self.resolveCoordinator() }
        }
        Task { await self.checkEncoderUpdate() }
    }
    
    deinit {
        monitor.cancel()
    }
    
    // MARK: - Coordinator Resolution
    
    private func resolveCoordinator() async {
        guard let coordURL = coordinatorURL else { return }
        var request = URLRequest(url: coordURL.appendingPathComponent("/api/status"))
        request.httpMethod = "GET"
        request.timeoutInterval = 5
        
        do {
            let (data, response) = try await session.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else { return }
            let status = try JSONDecoder().decode(CoordinatorStatusResponse.self, from: data)
            if status.singleDecoder, let first = status.decoders.first, let url = URL(string: first.endpoint) {
                effectiveDecoderURL = url
                useCoordinator = false
            } else {
                useCoordinator = true
            }
        } catch {
            logger.warning("Coordinator unreachable, falling back to direct decoder: \(error)")
            effectiveDecoderURL = decoderURL
        }
    }
    
    private func checkLocalDecoder() async {
        if let localURL = localDecoderURL {
            var request = URLRequest(url: localURL.appendingPathComponent("/health"))
            request.httpMethod = "GET"
            request.timeoutInterval = 2
            do {
                let (_, response) = try await session.data(for: request)
                localDecoderAvailable = (response as? HTTPURLResponse)?.statusCode == 200
                return
            } catch {
                logger.warning("Local decoder unreachable: \(error)")
                localDecoderAvailable = false
                return
            }
        }
        // Auto-scan common ports for a local decoder
        let ports = [8000, 8080, 9000]
        for port in ports {
            guard let url = URL(string: "http://localhost:\(port)/health") else { continue }
            var request = URLRequest(url: url)
            request.httpMethod = "GET"
            request.timeoutInterval = 1
            do {
                let (_, response) = try await session.data(for: request)
                if (response as? HTTPURLResponse)?.statusCode == 200 {
                    localDecoderURL = URL(string: "http://localhost:\(port)/decode")
                    localDecoderAvailable = true
                    return
                }
            } catch {
                continue
            }
        }
        localDecoderAvailable = false
    }
    
    private func getRequestURL() -> URL {
        if localDecoderAvailable, preferLocal, let localURL = localDecoderURL {
            return localURL
        }
        if useCoordinator, let coordURL = coordinatorURL {
            return coordURL.appendingPathComponent("/api/decode")
        }
        return effectiveDecoderURL
    }
    
    // MARK: - Encoder Update Check
    
    private func checkEncoderUpdate() async {
        guard let url = URL(string: "https://huggingface.co/\(hfRepo)/raw/main/models/production/encoder.onnx") else { return }
        var request = URLRequest(url: url)
        request.httpMethod = "HEAD"
        request.timeoutInterval = 5
        
        do {
            let (_, response) = try await session.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else { return }
            let remoteEtag = httpResponse.value(forHTTPHeaderField: "etag") ?? httpResponse.value(forHTTPHeaderField: "last-modified") ?? ""
            let cached = await encoder.getCachedEtag()
            if !remoteEtag.isEmpty, remoteEtag != cached {
                await encoder.cacheEtag(remoteEtag)
                await downloadEncoderUpdate()
            }
        } catch {
            // Offline or HF unavailable, bundled encoder is fine
        }
    }
    
    private func downloadEncoderUpdate() async {
        guard let url = URL(string: "https://huggingface.co/\(hfRepo)/resolve/main/models/production/encoder.onnx") else { return }
        var request = URLRequest(url: url)
        request.timeoutInterval = 30
        
        do {
            let (data, response) = try await session.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else { return }
            await encoder.updateEncoder(data)
        } catch {
            logger.warning("Failed to download encoder update, using bundled version: \(error)")
        }
    }
    
    // MARK: - Public Methods
    
    public func translate(
        text: String,
        from sourceLang: String,
        to targetLang: String,
        options: TranslationOptions? = nil
    ) async throws -> TranslationResponse {
        
        guard monitor.currentPath.status == .satisfied else {
            throw TranslationError.networkError(URLError(.notConnectedToInternet))
        }
        
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
        
        logger.info("Encoding text: \(text.prefix(50))...")
        let encodedData = try await encoder.encode(
            text: text,
            sourceLang: sourceLang,
            targetLang: targetLang
        )
        
        let response = try await sendToDecoder(
            encodedData: encodedData,
            targetLang: targetLang,
            options: options
        )
        
        addToCache(key: cacheKey, translation: response.translation)
        
        return response
    }
    
    public func translateBatch(
        texts: [String],
        from sourceLang: String,
        to targetLang: String,
        options: TranslationOptions? = nil
    ) async throws -> [TranslationResponse] {
        
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
        
        let targetURL = getRequestURL()
        var request = URLRequest(url: targetURL)
        request.httpMethod = "POST"
        request.httpBody = encodedData
        request.setValue(targetLang, forHTTPHeaderField: "X-Target-Language")
        request.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
        
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
        if translationCache.count >= maxCacheSize {
            if let firstKey = translationCache.keys.first {
                translationCache.removeValue(forKey: firstKey)
            }
        }
        
        translationCache[key] = translation
    }
}