// ios/UniversalTranslationSDK/Sources/TranslationEncoder.swift
import Foundation
import CoreML

public class TranslationEncoder {
    private let modelUrl: URL
    private let vocabularyManager: VocabularyManager
    private var encoderModel: MLModel?
    private var currentVocabulary: VocabularyPack?
    
    public init() throws {
        // Load model from bundle
        guard let modelUrl = Bundle.main.url(forResource: "UniversalEncoder", withExtension: "mlmodelc") else {
            throw TranslationError.modelNotFound
        }
        
        self.modelUrl = modelUrl
        self.vocabularyManager = VocabularyManager()
    }
    
    public func prepareTranslation(sourceLang: String, targetLang: String) async throws {
        // Download vocabulary if needed
        let vocabPack = try await vocabularyManager.getVocabularyForPair(
            source: sourceLang,
            target: targetLang
        )
        
        if vocabPack.needsDownload {
            try await downloadVocabulary(vocabPack)
        }
        
        // Load vocabulary
        currentVocabulary = vocabPack
        
        // Load model with configuration
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        encoderModel = try MLModel(contentsOf: modelUrl, configuration: config)
    }
    
    public func encode(text: String, sourceLang: String, targetLang: String) async throws -> Data {
        // Ensure prepared
        if currentVocabulary == nil {
            try await prepareTranslation(sourceLang: sourceLang, targetLang: targetLang)
        }
        
        guard let vocabulary = currentVocabulary else {
            throw TranslationError.vocabularyNotLoaded
        }
        
        // Tokenize
        let tokens = tokenize(text: text, sourceLang: sourceLang, vocabulary: vocabulary)
        
        // Create input
        let inputArray = try MLMultiArray(shape: [1, 128], dataType: .int32)
        for (i, token) in tokens.enumerated() {
            inputArray[i] = NSNumber(value: token)
        }
        
        // Run inference
        guard let model = encoderModel else {
            throw TranslationError.modelNotLoaded
        }
        
        let input = UniversalEncoderInput(input_ids: inputArray)
        let output = try model.prediction(input: input)
        
        // Compress output
        return try compressOutput(output.encoder_output)
    }
    
    private func tokenize(text: String, sourceLang: String, vocabulary: VocabularyPack) -> [Int32] {
        var tokens: [Int32] = []
        
        // Add language token
        if let langToken = vocabulary.tokens["<\(sourceLang)>"] {
            tokens.append(Int32(langToken))
        }
        
        // Tokenize words
        let words = text.lowercased().split(separator: " ")
        for word in words {
            if let tokenId = vocabulary.tokens[String(word)] {
                tokens.append(Int32(tokenId))
            } else {
                // Handle unknown
                tokens.append(Int32(vocabulary.tokens["<unk>"] ?? 1))
            }
        }
        
        // Add end token
        tokens.append(Int32(vocabulary.tokens["</s>"] ?? 3))
        
        // Pad to 128
        while tokens.count < 128 {
            tokens.append(Int32(vocabulary.tokens["<pad>"] ?? 0))
        }
        
        return Array(tokens.prefix(128))
    }
    
    private func compressOutput(_ output: MLMultiArray) throws -> Data {
        // Extract float data
        let count = output.count
        var floatArray = [Float32](repeating: 0, count: count)
        let ptr = output.dataPointer.bindMemory(to: Float32.self, capacity: count)
        floatArray = Array(UnsafeBufferPointer(start: ptr, count: count))
        
        // Quantize to Int8
        let maxVal = floatArray.map { abs($0) }.max() ?? 1.0
        let scale = 127.0 / maxVal
        
        let quantized = floatArray.map { Int8(round($0 * scale)) }
        
        // Create compressed data
        var compressedData = Data()
        
        // Add metadata
        withUnsafeBytes(of: Int32(output.shape[1].intValue)) { compressedData.append(contentsOf: $0) }
        withUnsafeBytes(of: Int32(output.shape[2].intValue)) { compressedData.append(contentsOf: $0) }
        withUnsafeBytes(of: scale) { compressedData.append(contentsOf: $0) }
        
        // Compress with zlib
        let quantizedData = Data(quantized)
        if let compressed = quantizedData.compressed(using: .lz4) {
            compressedData.append(compressed)
        }
        
        return compressedData
    }
}

// Translation Client
public class TranslationClient {
    private let encoder: TranslationEncoder
    private let decoderURL: URL
    private let session: URLSession
    
    public init(decoderURL: String) throws {
        self.encoder = try TranslationEncoder()
        self.decoderURL = URL(string: decoderURL)!
        
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        self.session = URLSession(configuration: config)
    }
    
    public func translate(
        text: String,
        from sourceLang: String,
        to targetLang: String
    ) async throws -> String {
        
        // Encode locally
        let encodedData = try await encoder.encode(
            text: text,
            sourceLang: sourceLang,
            targetLang: targetLang
        )
        
        // Send to decoder
        var request = URLRequest(url: decoderURL)
        request.httpMethod = "POST"
        request.httpBody = encodedData
        request.setValue(targetLang, forHTTPHeaderField: "X-Target-Language")
        request.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
        
        let (data, response) = try await session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw TranslationError.decodingFailed
        }
        
        let result = try JSONDecoder().decode(TranslationResponse.self, from: data)
        return result.translation
    }
}

// Usage in SwiftUI
struct ContentView: View {
    @State private var inputText = ""
    @State private var translatedText = ""
    @State private var isTranslating = false
    
    let translationClient = try! TranslationClient(
        decoderURL: "https://api.yourdomain.com/decode"
    )
    
    var body: some View {
        VStack {
            TextEditor(text: $inputText)
                .frame(height: 100)
                .border(Color.gray)
            
            Button("Translate to Spanish") {
                Task {
                    isTranslating = true
                    do {
                        translatedText = try await translationClient.translate(
                            text: inputText,
                            from: "en",
                            to: "es"
                        )
                    } catch {
                        translatedText = "Error: \(error)"
                    }
                    isTranslating = false
                }
            }
            .disabled(isTranslating)
            
            Text(translatedText)
                .padding()
        }
        .padding()
    }
}