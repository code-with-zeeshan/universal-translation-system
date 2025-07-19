// react-native/UniversalTranslationSDK/ios/UniversalTranslationModule.swift

import Foundation

@objc(UniversalTranslationModule)
class UniversalTranslationModule: RCTEventEmitter {
    
    private var encoder: TranslationEncoder?
    private var hasListeners = false
    private let encoderQueue = DispatchQueue(label: "com.universaltranslation.encoder", qos: .userInitiated)
    
    override init() {
        super.init()
    }
    
    @objc
    override static func requiresMainQueueSetup() -> Bool {
        return false
    }
    
    override func supportedEvents() -> [String]! {
        return ["vocabularyDownloadProgress"]
    }
    
    override func startObserving() {
        hasListeners = true
    }
    
    override func stopObserving() {
        hasListeners = false
    }
    
    @objc
    func initialize(_ resolve: @escaping RCTPromiseResolveBlock,
                   rejecter reject: @escaping RCTPromiseRejectBlock) {
        encoderQueue.async { [weak self] in
            do {
                if self?.encoder == nil {
                    self?.encoder = try TranslationEncoder()
                }
                resolve(nil)
            } catch {
                reject("INIT_ERROR", "Failed to initialize encoder", error)
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
                guard let encoder = self.encoder else {
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
    func encode(_ text: String,
                sourceLang: String,
                targetLang: String,
                resolver resolve: @escaping RCTPromiseResolveBlock,
                rejecter reject: @escaping RCTPromiseRejectBlock) {
        Task {
            do {
                guard let encoder = self.encoder else {
                    throw TranslationError.encoderNotInitialized
                }
                
                let encoded = try await encoder.encode(
                    text: text,
                    sourceLang: sourceLang,
                    targetLang: targetLang
                )
                
                // Convert to base64 for React Native bridge
                let base64 = encoded.base64EncodedString()
                resolve(base64)
            } catch {
                reject("ENCODE_ERROR", error.localizedDescription, error)
            }
        }
    }
    
    @objc
    func getAvailableVocabularies(_ resolve: @escaping RCTPromiseResolveBlock,
                                 rejecter reject: @escaping RCTPromiseRejectBlock) {
        Task {
            do {
                let vocabManager = VocabularyManager()
                let vocabs = try await vocabManager.getAvailableVocabularies()
                
                let vocabArray = vocabs.map { vocab in
                    return [
                        "name": vocab.name,
                        "languages": vocab.languages,
                        "sizeMB": vocab.sizeMB,
                        "isDownloaded": !vocab.needsDownload,
                        "version": vocab.version
                    ]
                }
                
                resolve(vocabArray)
            } catch {
                reject("VOCAB_ERROR", error.localizedDescription, error)
            }
        }
    }
    
    @objc
    func downloadVocabulary(_ name: String,
                           resolver resolve: @escaping RCTPromiseResolveBlock,
                           rejecter reject: @escaping RCTPromiseRejectBlock) {
        Task {
            do {
                let vocabManager = VocabularyManager()
                
                try await vocabManager.downloadVocabulary(name: name) { [weak self] progress in
                    guard self?.hasListeners == true else { return }
                    
                    self?.sendEvent(withName: "vocabularyDownloadProgress", body: [
                        "name": name,
                        "progress": progress
                    ])
                }
                
                resolve(nil)
            } catch {
                reject("DOWNLOAD_ERROR", error.localizedDescription, error)
            }
        }
    }
    
    @objc
    func deleteVocabulary(_ name: String,
                         resolver resolve: @escaping RCTPromiseResolveBlock,
                         rejecter reject: @escaping RCTPromiseRejectBlock) {
        do {
            let vocabManager = VocabularyManager()
            try vocabManager.deleteVocabulary(packName: name)
            resolve(nil)
        } catch {
            reject("DELETE_ERROR", error.localizedDescription, error)
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
                guard let encoder = self.encoder else {
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
    func clearCache(_ resolve: @escaping RCTPromiseResolveBlock,
                   rejecter reject: @escaping RCTPromiseRejectBlock) {
        // Clear any caches
        resolve(nil)
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