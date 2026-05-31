// ios/UniversalTranslationSDK/Sources/SwiftUI/TranslationEnvironment.swift

import SwiftUI

// Environment key for translation client
private struct TranslationClientKey: EnvironmentKey {
    static let defaultValue: TranslationClient? = nil
}

extension EnvironmentValues {
    public var translationClient: TranslationClient? {
        get { self[TranslationClientKey.self] }
        set { self[TranslationClientKey.self] = newValue }
    }
}

// View modifier for easy setup
public struct TranslationEnabledModifier: ViewModifier {
    let client: TranslationClient
    @State private var isInitialized = false
    @State private var initError: Error?
    
    public init(decoderURL: String = "https://api.yourdomain.com/decode") {
        do {
            self.client = try TranslationClient(decoderURL: decoderURL)
        } catch {
            fatalError("Failed to create TranslationClient: \(error)")
        }
    }
    
    public func body(content: Content) -> some View {
        content
            .environment(\.translationClient, client)
            .task {
                do {
                    try await client.encoder.initialize()
                    
                    // Enable background translation
                    await client.enableBackgroundTranslation()
                    
                    // Prefetch vocabularies
                    Task {
                        await client.vocabularyManager.prefetchVocabulariesForUserLanguages()
                    }
                    
                    isInitialized = true
                } catch {
                    initError = error
                    print("Failed to initialize translation: \(error)")
                }
            }
            .overlay(alignment: .top) {
                if let error = initError {
                    Text("Translation initialization failed: \(error.localizedDescription)")
                        .padding()
                        .background(Color.red.opacity(0.8))
                        .foregroundColor(.white)
                        .cornerRadius(8)
                        .padding()
                }
            }
    }
}

extension View {
    public func translationEnabled(decoderURL: String = "https://api.yourdomain.com/decode") -> some View {
        self.modifier(TranslationEnabledModifier(decoderURL: decoderURL))
    }
}

// Convenience translation view
public struct QuickTranslationButton: View {
    @Environment(\.translationClient) var translator
    let text: String
    let sourceLang: String
    let targetLang: String
    @State private var translation: String?
    @State private var isTranslating = false
    
    public var body: some View {
        Button(action: translate) {
            if isTranslating {
                ProgressView()
            } else if let translation = translation {
                Text(translation)
            } else {
                Label("Translate", systemImage: "translate")
            }
        }
        .disabled(isTranslating || translator == nil)
    }
    
    private func translate() {
        guard let translator = translator else { return }
        
        isTranslating = true
        Task {
            do {
                let response = try await translator.translate(
                    text: text,
                    from: sourceLang,
                    to: targetLang
                )
                translation = response.translation
            } catch {
                print("Translation failed: \(error)")
            }
            isTranslating = false
        }
    }
}