// ios/UniversalTranslationSDK/Sources/Analytics.swift

import Foundation
import OSLog

public enum TranslationErrorCode: String {
    case networkError = "NETWORK_ERROR"
    case modelNotFound = "MODEL_NOT_FOUND"
    case vocabularyNotLoaded = "VOCABULARY_NOT_LOADED"
    case encodingFailed = "ENCODING_FAILED"
    case decodingFailed = "DECODING_FAILED"
    case invalidLanguage = "INVALID_LANGUAGE"
    case rateLimited = "RATE_LIMITED"
}

public protocol AnalyticsProvider {
    func trackEvent(_ event: String, parameters: [String: Any])
}

public class TranslationAnalytics {
    private let provider: AnalyticsProvider?
    private let logger = Logger(subsystem: "com.universaltranslation", category: "Analytics")
    
    public init(provider: AnalyticsProvider? = nil) {
        self.provider = provider
    }
    
    public func trackTranslation(
        sourceLang: String,
        targetLang: String,
        textLength: Int,
        duration: TimeInterval
    ) {
        let parameters: [String: Any] = [
            "source_lang": sourceLang,
            "target_lang": targetLang,
            "text_length": textLength,
            "duration_ms": Int(duration * 1000),
            "platform": "iOS",
            "os_version": ProcessInfo.processInfo.operatingSystemVersionString
        ]
        
        provider?.trackEvent("translation_completed", parameters: parameters)
        logger.info("Translation tracked: \(sourceLang)->\(targetLang), \(textLength) chars, \(duration)s")
    }
    
    public func trackError(error: TranslationErrorCode, context: [String: Any]) {
        var parameters = context
        parameters["error_code"] = error.rawValue
        parameters["platform"] = "iOS"
        
        provider?.trackEvent("translation_error", parameters: parameters)
        logger.error("Translation error: \(error.rawValue), context: \(context)")
    }
}

// Update TranslationClient to use analytics
extension TranslationClient {
    private static let analytics = TranslationAnalytics()
    
    // Update translate method to include analytics
    func translateWithAnalytics(
        text: String,
        from sourceLang: String,
        to targetLang: String,
        options: TranslationOptions? = nil
    ) async throws -> TranslationResponse {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        do {
            let response = try await translate(
                text: text,
                from: sourceLang,
                to: targetLang,
                options: options
            )
            
            let duration = CFAbsoluteTimeGetCurrent() - startTime
            Self.analytics.trackTranslation(
                sourceLang: sourceLang,
                targetLang: targetLang,
                textLength: text.count,
                duration: duration
            )
            
            return response
        } catch {
            Self.analytics.trackError(
                error: mapErrorToCode(error),
                context: [
                    "source_lang": sourceLang,
                    "target_lang": targetLang,
                    "error_message": error.localizedDescription
                ]
            )
            throw error
        }
    }
    
    private func mapErrorToCode(_ error: Error) -> TranslationErrorCode {
        // Map your errors to error codes
        if error is URLError {
            return .networkError
        } else if let translationError = error as? TranslationError {
            switch translationError {
            case .modelNotFound: return .modelNotFound
            case .vocabularyNotLoaded: return .vocabularyNotLoaded
            case .encodingFailed: return .encodingFailed
            case .decodingFailed: return .decodingFailed
            case .unsupportedLanguage: return .invalidLanguage
            default: return .encodingFailed
            }
        }
        return .encodingFailed
    }
}