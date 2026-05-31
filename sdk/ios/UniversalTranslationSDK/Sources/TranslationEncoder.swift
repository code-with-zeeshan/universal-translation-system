// ios/UniversalTranslationSDK/Sources/TranslationEncoder.swift

import Foundation
import CoreML
import Compression
import OSLog
import Accelerate
import BackgroundTasks

// MARK: - Error Types

public enum TranslationError: LocalizedError {
    case modelNotFound
    case vocabularyNotLoaded
    case modelNotLoaded
    case tokenizationFailed
    case encodingFailed
    case decodingFailed
    case compressionFailed
    case networkError(Error)
    case invalidInput
    case unsupportedLanguage(String)
    
    public var errorDescription: String? {
        switch self {
        case .modelNotFound:
            return "Translation model not found in bundle"
        case .vocabularyNotLoaded:
            return "Vocabulary not loaded"
        case .modelNotLoaded:
            return "Model not loaded"
        case .tokenizationFailed:
            return "Failed to tokenize input text"
        case .encodingFailed:
            return "Failed to encode text"
        case .decodingFailed:
            return "Failed to decode translation"
        case .compressionFailed:
            return "Failed to compress encoder output"
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        case .invalidInput:
            return "Invalid input text"
        case .unsupportedLanguage(let lang):
            return "Unsupported language: \(lang)"
        }
    }
}

// MARK: - Logger

private let logger = Logger(subsystem: "com.universaltranslation.sdk", category: "TranslationEncoder")

// Add performance monitoring
private class PerformanceMonitor {
    private var metrics: [String: [TimeInterval]] = [:]
    private let queue = DispatchQueue(label: "com.translation.performance", attributes: .concurrent)
    
    func measure<T>(_ operation: String, block: () throws -> T) rethrows -> T {
        let startTime = CFAbsoluteTimeGetCurrent()
        defer {
            let duration = CFAbsoluteTimeGetCurrent() - startTime
            queue.async(flags: .barrier) {
                if self.metrics[operation] == nil {
                    self.metrics[operation] = []
                }
                self.metrics[operation]?.append(duration)
                
                if duration > 1.0 {
                    logger.warning("\(operation) took \(duration)s")
                }
            }
        }
        return try block()
    }
    
    func getMetrics() -> [String: [TimeInterval]] {
        queue.sync { metrics }
    }
}

// MARK: - Translation Encoder

@available(iOS 15.0, macOS 12.0, *)
public actor TranslationEncoder {
    private let performanceMonitor = PerformanceMonitor() // performance monitor
    private var isWarmedUp = false //  warmup flag
    private let modelURL: URL
    private let vocabularyManager: VocabularyManager
    private var encoderModel: MLModel?
    private var currentVocabulary: VocabularyPack?
    private let modelQueue = DispatchQueue(label: "com.universaltranslation.model", qos: .userInitiated)
    
    // Cache for loaded models
    private var modelCache: [String: MLModel] = [:]

    public func initialize() async throws {
        if encoderModel != nil { return }
        
        try await loadModel()
        
        // Warm up model in background
        Task.detached(priority: .background) { [weak self] in
            await self?.warmupModel()
        }
    }

    // Add warmup method
    private func warmupModel() async {
        guard !isWarmedUp else { return }
        
        do {
            logger.info("Starting model warmup...")
            let startTime = CFAbsoluteTimeGetCurrent()
            
            // Run dummy inference
            _ = try await encode(
                text: "test",
                sourceLang: "en",
                targetLang: "es"
            )
            
            isWarmedUp = true
            let warmupTime = CFAbsoluteTimeGetCurrent() - startTime
            logger.info("Model warmup completed in \(warmupTime)s")
        } catch {
            logger.warning("Model warmup failed: \(error)")
        }
    }
    
    public init() throws {
        // Try multiple locations for the model
        if let bundleURL = Bundle.main.url(forResource: "UniversalEncoder", withExtension: "mlmodelc") {
            self.modelURL = bundleURL
        } else if let frameworkBundle = Bundle(for: VocabularyManager.self).url(forResource: "UniversalEncoder", withExtension: "mlmodelc") {
            self.modelURL = frameworkBundle
        } else {
            // Try to find in app's Documents directory (for downloaded models)
            let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            let modelPath = documentsPath.appendingPathComponent("Models/UniversalEncoder.mlmodelc")
            
            if FileManager.default.fileExists(atPath: modelPath.path) {
                self.modelURL = modelPath
            } else {
                logger.error("Model not found in any location")
                throw TranslationError.modelNotFound
            }
        }
        
        self.vocabularyManager = VocabularyManager()
        logger.info("TranslationEncoder initialized with model at: \(self.modelURL.path)")
    }
    
    public func prepareTranslation(sourceLang: String, targetLang: String) async throws {
        logger.info("Preparing translation for \(sourceLang) -> \(targetLang)")
        
        // Validate languages
        guard VocabularyManager.supportedLanguages.contains(sourceLang) else {
            throw TranslationError.unsupportedLanguage(sourceLang)
        }
        guard VocabularyManager.supportedLanguages.contains(targetLang) else {
            throw TranslationError.unsupportedLanguage(targetLang)
        }
        
        // Get vocabulary pack
        let vocabPack = try await vocabularyManager.getVocabularyForPair(
            source: sourceLang,
            target: targetLang
        )
        
        // Download if needed
        if vocabPack.needsDownload {
            logger.info("Downloading vocabulary pack: \(vocabPack.name)")
            try await vocabularyManager.downloadVocabulary(vocabPack)
        }
        
        // Load vocabulary
        currentVocabulary = try await vocabularyManager.loadVocabulary(from: vocabPack.localPath)
        
        // Load model if not already loaded
        if encoderModel == nil {
            try await loadModel()
        }
        
        logger.info("Translation prepared successfully")
    }
    
    // Update loadModel with optimizations
    private func loadModel() async throws {
        let config = MLModelConfiguration()
        
        // Optimize for Neural Engine
        #if os(iOS)
        if #available(iOS 16.0, *) { // iOS-specific optimizations
            config.modelDisplayName = "Universal Encoder"
            config.computeUnits = .cpuAndNeuralEngine
            config.allowLowPrecisionAccumulationOnGPU = true

            // Enable model caching
            config.parameters = [
                .enablePerformanceShaping: true,
                .profileDuringCompilation: true
            ]
        } else {
            config.computeUnits = .all
        }
        #else
        config.computeUnits = .cpuAndNeuralEngine
        #endif

        config.allowLowPrecisionAccumulationOnGPU = true
        
        encoderModel = try await performanceMonitor.measure("loadModel") {
            try await Task {
                try MLModel(contentsOf: modelURL, configuration: config)
            }.value
        }
        
        logger.info("Model loaded successfully with configuration: \(config)")
    }
    
    // Update encode method with performance monitoring
    public func encode(text: String, sourceLang: String, targetLang: String) async throws -> Data {
        try await performanceMonitor.measure("encode") {
            // Existing encode implementation...
            if !isWarmedUp {
                logger.info("Encoding without warmup")
            }
            // Validate input
            let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmedText.isEmpty else {
                throw TranslationError.invalidInput
            }
        
            // Ensure prepared
            if currentVocabulary == nil {
                try await prepareTranslation(sourceLang: sourceLang, targetLang: targetLang)
            }
        
            guard let vocabulary = currentVocabulary else {
                throw TranslationError.vocabularyNotLoaded
            }
        
            guard let model = encoderModel else {
                throw TranslationError.modelNotLoaded
            }
        
            // Tokenize
            let tokens = try tokenize(text: trimmedText, sourceLang: sourceLang, vocabulary: vocabulary)
        
            // Create input array
            let inputArray = try MLMultiArray(shape: [1, 128], dataType: .int32)
            for (i, token) in tokens.enumerated() {
                inputArray[i] = NSNumber(value: token)
            }
        
            // Create model input
            let input = try createModelInput(inputArray: inputArray)
        
            // Run inference
            let output = try await Task {
                try model.prediction(from: input)
            }.value
        
            // Extract and compress output
            let outputArray = try extractEncoderOutput(from: output)
            return try compressOutput(outputArray)
        }
    }   

    // Add method to get performance metrics
    public func getPerformanceMetrics() -> [String: [TimeInterval]] {
        performanceMonitor.getMetrics()
    } 
    
    private func tokenize(text: String, sourceLang: String, vocabulary: VocabularyPack) throws -> [Int32] {
        var tokens: [Int32] = []
        
        // Add BOS token
        tokens.append(Int32(vocabulary.specialTokens["<s>"] ?? 2))
        
        // Add language token
        let langToken = "<\(sourceLang)>"
        if let langTokenId = vocabulary.specialTokens[langToken] {
            tokens.append(Int32(langTokenId))
        }
        
        // Tokenize text (simple whitespace tokenization for demo)
        // In production, use a proper tokenizer like SentencePiece
        let words = text.lowercased().components(separatedBy: .whitespacesAndNewlines)
        
        for word in words where !word.isEmpty {
            if let tokenId = vocabulary.tokens[word] {
                tokens.append(Int32(tokenId))
            } else {
                // Handle subword tokenization for unknown words
                let subwordTokens = tokenizeUnknownWord(word, vocabulary: vocabulary)
                tokens.append(contentsOf: subwordTokens)
            }
        }
        
        // Add EOS token
        tokens.append(Int32(vocabulary.specialTokens["</s>"] ?? 3))
        
        // Pad or truncate to 128
        if tokens.count < 128 {
            let padToken = Int32(vocabulary.specialTokens["<pad>"] ?? 0)
            tokens.append(contentsOf: Array(repeating: padToken, count: 128 - tokens.count))
        } else if tokens.count > 128 {
            tokens = Array(tokens.prefix(127))
            tokens.append(Int32(vocabulary.specialTokens["</s>"] ?? 3))
        }
        
        return tokens
    }
    
    private func tokenizeUnknownWord(_ word: String, vocabulary: VocabularyPack) -> [Int32] {
        // Simple subword tokenization
        var subwordTokens: [Int32] = []
        let unkToken = Int32(vocabulary.specialTokens["<unk>"] ?? 1)
        
        // Try to find subword matches
        var position = 0
        while position < word.count {
            var found = false
            
            // Try different subword lengths
            for length in stride(from: min(word.count - position, 10), to: 0, by: -1) {
                let startIndex = word.index(word.startIndex, offsetBy: position)
                let endIndex = word.index(startIndex, offsetBy: length)
                let subword = "##" + String(word[startIndex..<endIndex])
                
                if let subwordId = vocabulary.subwords[subword] {
                    subwordTokens.append(Int32(subwordId))
                    position += length
                    found = true
                    break
                }
            }
            
            if !found {
                subwordTokens.append(unkToken)
                position += 1
            }
        }
        
        return subwordTokens.isEmpty ? [unkToken] : subwordTokens
    }
    
    private func createModelInput(inputArray: MLMultiArray) throws -> MLFeatureProvider {
        // Create a dictionary of features
        let inputFeatures: [String: MLFeatureValue] = [
            "input_ids": MLFeatureValue(multiArray: inputArray)
        ]
        
        return try MLDictionaryFeatureProvider(dictionary: inputFeatures)
    }
    
    private func extractEncoderOutput(from output: MLFeatureProvider) throws -> MLMultiArray {
        // Try different possible output names
        let possibleNames = ["encoder_output", "output", "embeddings", "hidden_states"]
        
        for name in possibleNames {
            if let outputValue = output.featureValue(for: name),
               let multiArray = outputValue.multiArrayValue {
                return multiArray
            }
        }
        
        throw TranslationError.encodingFailed
    }
    
    private func compressOutput(_ output: MLMultiArray) throws -> Data {
        // Get output shape
        let shape = output.shape.map { $0.intValue }
        guard shape.count >= 2 else {
            throw TranslationError.compressionFailed
        }
        
        let sequenceLength = shape[shape.count - 2]
        let hiddenSize = shape[shape.count - 1]
        let totalElements = output.count
        
        // Extract float data
        var floatArray = [Float32](repeating: 0, count: totalElements)
        
        if output.dataType == .float32 {
            let dataPointer = output.dataPointer.bindMemory(to: Float32.self, capacity: totalElements)
            floatArray = Array(UnsafeBufferPointer(start: dataPointer, count: totalElements))
        } else if output.dataType == .double {
            let dataPointer = output.dataPointer.bindMemory(to: Double.self, capacity: totalElements)
            let doubleArray = Array(UnsafeBufferPointer(start: dataPointer, count: totalElements))
            floatArray = doubleArray.map { Float32($0) }
        }
        
        // Quantize to Int8
        let maxAbsValue = floatArray.map { abs($0) }.max() ?? 1.0
        let scale = maxAbsValue > 0 ? 127.0 / maxAbsValue : 1.0
        
        let quantized = floatArray.map { value -> Int8 in
            let scaled = value * scale
            return Int8(max(-128, min(127, round(scaled))))
        }
        
        // Create output data
        var outputData = Data()
        
        // Add metadata (16 bytes)
        outputData.append(contentsOf: withUnsafeBytes(of: Int32(sequenceLength)) { Array($0) })
        outputData.append(contentsOf: withUnsafeBytes(of: Int32(hiddenSize)) { Array($0) })
        outputData.append(contentsOf: withUnsafeBytes(of: Float32(scale)) { Array($0) })
        outputData.append(contentsOf: withUnsafeBytes(of: Int32(0)) { Array($0) }) // Reserved
        
        // Compress quantized data
        let quantizedData = Data(quantized)
        
        // Use compression
        guard let compressedData = quantizedData.compressed(using: .lz4) else {
            throw TranslationError.compressionFailed
        }
        
        outputData.append(compressedData)
        
        logger.info("Compressed output: \(floatArray.count * 4) bytes -> \(outputData.count) bytes")
        
        return outputData
    }
    
    // MARK: - Public Utilities
    
    public func getSupportedLanguages() -> [String] {
        return VocabularyManager.supportedLanguages
    }
    
    public func getMemoryUsage() -> Int {
        // Estimate memory usage
        var totalMemory = 0
        
        // Model memory (rough estimate)
        if encoderModel != nil {
            totalMemory += 100 * 1024 * 1024 // ~100MB for model
        }
        
        // Vocabulary memory
        if let vocab = currentVocabulary {
            totalMemory += (vocab.tokens.count + vocab.subwords.count) * 50 // Rough estimate
        }
        
        return totalMemory
    }
}

// MARK: - Extensions

extension Data {
    func compressed(using algorithm: NSData.CompressionAlgorithm) -> Data? {
        return (self as NSData).compressed(using: algorithm) as Data?
    }
}