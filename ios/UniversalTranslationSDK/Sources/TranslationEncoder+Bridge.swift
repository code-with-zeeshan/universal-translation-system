// ios/UniversalTranslationSDK/Sources/TranslationEncoder+Bridge.swift

import Foundation
import OSLog

// MARK: - Swift wrapper for C++ encoder

@available(iOS 15.0, macOS 12.0, *)
extension TranslationEncoder {
    
    // Alternative implementation using C++ bridge
    private class EncoderBridgeWrapper {
        private let bridge: UniversalEncoderBridge
        private let logger = Logger(subsystem: "com.universaltranslation.sdk", category: "EncoderBridge")
        
        init(modelPath: URL) throws {
            var error: NSError?
            guard let bridge = UniversalEncoderBridge(modelPath: modelPath.path, error: &error) else {
                throw error ?? TranslationError.modelNotFound
            }
            self.bridge = bridge
        }
        
        func loadVocabulary(path: String) throws {
            var error: NSError?
            guard bridge.loadVocabulary(path, error: &error) else {
                throw error ?? TranslationError.vocabularyNotLoaded
            }
        }
        
        func encode(text: String, sourceLang: String, targetLang: String) throws -> Data {
            var error: NSError?
            guard let data = bridge.encodeText(text, sourceLang: sourceLang, targetLang: targetLang, error: &error) else {
                throw error ?? TranslationError.encodingFailed
            }
            return data
        }
        
        func getSupportedLanguages() -> [String] {
            return bridge.getSupportedLanguages() as? [String] ?? []
        }
        
        func getMemoryUsage() -> Int {
            return Int(bridge.getMemoryUsage())
        }
    }
    
    // Add property to track which implementation to use
    private var useCppBridge: Bool {
        // Check if C++ model is available
        let cppModelPath = modelURL.deletingPathExtension().appendingPathExtension("onnx")
        return FileManager.default.fileExists(atPath: cppModelPath.path)
    }
    
    // Alternative encode method using C++ bridge
    private func encodeUsingCppBridge(text: String, sourceLang: String, targetLang: String) async throws -> Data {
        logger.info("Using C++ bridge for encoding")
        
        // Create bridge wrapper if needed
        let bridge = try EncoderBridgeWrapper(modelPath: modelURL)
        
        // Load vocabulary if needed
        if let vocabPath = currentVocabulary?.localPath {
            try bridge.loadVocabulary(path: vocabPath)
        }
        
        // Encode using C++ implementation
        return try bridge.encode(text: text, sourceLang: sourceLang, targetLang: targetLang)
    }
}

// MARK: - Unified Encoding Interface

@available(iOS 15.0, macOS 12.0, *)
extension TranslationEncoder {
    
    public func encodeWithBestAvailableMethod(
        text: String,
        sourceLang: String,
        targetLang: String
    ) async throws -> Data {
        // Try CoreML first (fastest on iOS)
        if encoderModel != nil {
            do {
                return try await encode(text: text, sourceLang: sourceLang, targetLang: targetLang)
            } catch {
                logger.warning("CoreML encoding failed, trying C++ bridge: \(error)")
            }
        }
        
        // Fallback to C++ bridge
        if useCppBridge {
            return try await encodeUsingCppBridge(
                text: text,
                sourceLang: sourceLang,
                targetLang: targetLang
            )
        }
        
        throw TranslationError.encodingFailed
    }
}