// react-native/UniversalTranslationSDK/ios/UniversalTranslationModule.swift

import Foundation

@objc(UniversalTranslationModule)
class UniversalTranslationModule: RCTEventEmitter {
    
    private var translationClient: TranslationClient?
    private var hasListeners = false
    private let clientQueue = DispatchQueue(label: "com.universaltranslation.client", qos: .userInitiated)
    
    override init() {
        super.init()
    }
    
    @objc
    override static func requiresMainQueueSetup() -> Bool {
        return false
    }
    
    override func supportedEvents() -> [String]! {
        return ["vocabularyDownloadProgress", "translationProgress"]
    }
    
    override func startObserving() {
        hasListeners = true
    }
    
    override func stopObserving() {
        hasListeners = false
    }
    
    @objc
    func initialize(_ decoderURL: String,
                   resolver resolve: @escaping RCTPromiseResolveBlock,
                   rejecter reject: @escaping RCTPromiseRejectBlock) {
        Task {
            do {
                self.translationClient = try TranslationClient(decoderURL: decoderURL)
                resolve(nil)
            } catch {
                reject("INIT_ERROR", "Failed to initialize: \(error.localizedDescription)", error)
            }
        }
    }
    
    @objc
    func translate(_ text: String,
                  sourceLang: String,
                  targetLang: String,
                  resolver resolve: @escaping RCTPromiseResolveBlock,
                  rejecter reject: @escaping RCTPromiseRejectBlock) {
        Task {
            do {
                guard let client = self.translationClient else {
                    throw TranslationError.encoderNotInitialized
                }
                
                let response = try await client.translate(
                    text: text,
                    from: sourceLang,
                    to: targetLang
                )
                
                let result: [String: Any] = [
                    "translation": response.translation,
                    "targetLang": response.targetLang,
                    "confidence": response.confidence ?? 1.0
                ]
                
                resolve(result)
            } catch {
                reject("TRANSLATION_ERROR", error.localizedDescription, error)
            }
        }
    }
    
    @objc
    func prepareTranslation(_ sourceLang: String,
                           targetLang: String,
                           resolver resolve: @escaping RCTPromiseResolveBlock,
                           rejecter reject: @escaping RCTPromiseRejectBlock) {
        Task {
            do {
                guard let encoder = await self.translationClient?.encoder else {
                    throw TranslationError.encoderNotInitialized
                }
                
                try await encoder.prepareTranslation(sourceLang: sourceLang, targetLang: targetLang)
                resolve(nil)
            } catch {
                reject("PREPARE_ERROR", error.localizedDescription, error)
            }
        }
    }
    
    @objc
    func getVocabularyForPair(_ sourceLang: String,
                             targetLang: String,
                             resolver resolve: @escaping RCTPromiseResolveBlock,
                             rejecter reject: @escaping RCTPromiseRejectBlock) {
        Task {
            do {
                let vocabManager = VocabularyManager()
                let pack = try await vocabManager.getVocabularyForPair(
                    source: sourceLang,
                    target: targetLang
                )
                
                let result: [String: Any] = [
                    "name": pack.name,
                    "languages": pack.languages,
                    "downloadUrl": pack.downloadURL,
                    "localPath": pack.localPath,
                    "sizeMb": pack.sizeMB,
                    "version": pack.version,
                    "needsDownload": pack.needsDownload
                ]
                
                resolve(result)
            } catch {
                reject("VOCAB_ERROR", error.localizedDescription, error)
            }
        }
    }
    
    @objc
    func downloadVocabularyPacks(_ languages: [String],
                                resolver resolve: @escaping RCTPromiseResolveBlock,
                                rejecter reject: @escaping RCTPromiseRejectBlock) {
        Task {
            do {
                let vocabManager = VocabularyManager()
                
                // Download packs for each language
                for language in languages {
                    if let pack = try? await vocabManager.getVocabularyForPair(
                        source: language,
                        target: language
                    ), pack.needsDownload {
                        try await vocabManager.downloadVocabulary(pack)
                        
                        // Send progress event if listeners are active
                        if hasListeners {
                            sendEvent(withName: "vocabularyDownloadProgress", body: [
                                "language": language,
                                "progress": 100
                            ])
                        }
                    }
                }
                
                resolve(nil)
            } catch {
                reject("DOWNLOAD_ERROR", error.localizedDescription, error)
            }
        }
    }
    
    @objc
    func getSupportedLanguages(_ resolve: @escaping RCTPromiseResolveBlock,
                              rejecter reject: @escaping RCTPromiseRejectBlock) {
        let languages = LanguageInfo.supportedLanguages.map { lang in
            return [
                "code": lang.code,
                "name": lang.name,
                "nativeName": lang.nativeName,
                "isRTL": lang.isRTL
            ]
        }
        
        resolve(languages)
    }
    
    @objc
    func getMemoryUsage(_ resolve: @escaping RCTPromiseResolveBlock,
                       rejecter reject: @escaping RCTPromiseRejectBlock) {
        Task {
            do {
                guard let encoder = await self.translationClient?.encoder else {
                    throw TranslationError.encoderNotInitialized
                }
                
                let memoryUsage = await encoder.getMemoryUsage()
                resolve(memoryUsage)
            } catch {
                reject("MEMORY_ERROR", error.localizedDescription, error)
            }
        }
    }
    
    @objc
    func clearTranslationCache(_ resolve: @escaping RCTPromiseResolveBlock,
                              rejecter reject: @escaping RCTPromiseRejectBlock) {
        Task {
            await self.translationClient?.clearCache()
            resolve(nil)
        }
    }
}

// MARK: - Objective-C Bridge

extension UniversalTranslationModule {
    @objc
    override static func moduleName() -> String! {
        return "UniversalTranslationModule"
    }
}

// Custom errors
enum TranslationError: LocalizedError {
    case encoderNotInitialized
    
    var errorDescription: String? {
        switch self {
        case .encoderNotInitialized:
            return "Translation encoder not initialized"
        }
    }
}