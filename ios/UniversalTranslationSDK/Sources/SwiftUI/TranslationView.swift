// ios/UniversalTranslationSDK/Sources/SwiftUI/TranslationView.swift

import SwiftUI

@available(iOS 15.0, macOS 12.0, *)
public struct TranslationView: View {
    @State private var inputText = ""
    @State private var translatedText = ""
    @State private var isTranslating = false
    @State private var errorMessage: String?
    @State private var sourceLang = "en"
    @State private var targetLang = "es"
    @State private var showingLanguagePicker = false
    @State private var pickingSource = true
    
    private let translationClient: TranslationClient
    
    public init(decoderURL: String = "https://api.yourdomain.com/decode") {
        do {
            self.translationClient = try TranslationClient(decoderURL: decoderURL)
        } catch {
            fatalError("Failed to initialize TranslationClient: \(error)")
        }
    }
    
    public var body: some View {
        NavigationView {
            VStack(spacing: 16) {
                // Language selector
                HStack {
                    LanguageButton(
                        language: sourceLang,
                        title: "From"
                    ) {
                        pickingSource = true
                        showingLanguagePicker = true
                    }
                    
                    Button(action: swapLanguages) {
                        Image(systemName: "arrow.left.arrow.right")
                            .font(.title2)
                    }
                    .disabled(isTranslating)
                    
                    LanguageButton(
                        language: targetLang,
                        title: "To"
                    ) {
                        pickingSource = false
                        showingLanguagePicker = true
                    }
                }
                .padding(.horizontal)
                
                // Input text
                VStack(alignment: .leading) {
                    Text("Enter text")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    TextEditor(text: $inputText)
                        .padding(8)
                        .background(Color(.systemGray6))
                        .cornerRadius(8)
                        .frame(minHeight: 100, maxHeight: 200)
                }
                .padding(.horizontal)
                
                // Translate button
                Button(action: translate) {
                    HStack {
                        if isTranslating {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle())
                                .scaleEffect(0.8)
                        } else {
                            Image(systemName: "translate")
                        }
                        Text("Translate")
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || isTranslating)
                
                // Error message
                if let error = errorMessage {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.red)
                        Text(error)
                            .font(.caption)
                            .foregroundColor(.red)
                    }
                    .padding(.horizontal)
                }
                
                // Translation result
                if !translatedText.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Translation")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            
                            Spacer()
                            
                            Button(action: copyTranslation) {
                                Image(systemName: "doc.on.doc")
                                    .font(.caption)
                            }
                        }
                        
                        Text(translatedText)
                            .padding()
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color(.systemGray6))
                            .cornerRadius(8)
                            .textSelection(.enabled)
                    }
                    .padding(.horizontal)
                }
                
                Spacer()
            }
            .navigationTitle("Universal Translation")
            .sheet(isPresented: $showingLanguagePicker) {
                LanguagePickerView(
                    selectedLanguage: pickingSource ? $sourceLang : $targetLang,
                    excludeLanguage: pickingSource ? targetLang : sourceLang
                )
            }
        }
    }
    
    private func translate() {
        isTranslating = true
        errorMessage = nil
        
        Task {
            do {
                let response = try await translationClient.translate(
                    text: inputText,
                    from: sourceLang,
                    to: targetLang
                )
                
                await MainActor.run {
                    self.translatedText = response.translation
                    self.isTranslating = false
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = error.localizedDescription
                    self.isTranslating = false
                }
            }
        }
    }
    
    private func swapLanguages() {
        let temp = sourceLang
        sourceLang = targetLang
        targetLang = temp
        
        if !translatedText.isEmpty {
            inputText = translatedText
            translatedText = ""
        }
    }
    
    private func copyTranslation() {
        #if os(iOS)
        UIPasteboard.general.string = translatedText
        #elseif os(macOS)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(translatedText, forType: .string)
        #endif
    }
}

// MARK: - Supporting Views

@available(iOS 15.0, macOS 12.0, *)
struct LanguageButton: View {
    let language: String
    let title: String
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Text(title)
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text(LanguageInfo.supportedLanguages.first { $0.code == language }?.name ?? language)
                    .font(.headline)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 8)
            .background(Color(.systemGray6))
            .cornerRadius(8)
        }
        .buttonStyle(.plain)
    }
}

@available(iOS 15.0, macOS 12.0, *)
struct LanguagePickerView: View {
    @Binding var selectedLanguage: String
    let excludeLanguage: String
    @Environment(\.dismiss) var dismiss
    
    var body: some View {
        NavigationView {
            List(LanguageInfo.supportedLanguages.filter { $0.code != excludeLanguage }) { language in
                Button(action: {
                    selectedLanguage = language.code
                    dismiss()
                }) {
                    HStack {
                        VStack(alignment: .leading) {
                            Text(language.name)
                                .font(.headline)
                            Text(language.nativeName)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        Spacer()
                        
                        if language.code == selectedLanguage {
                            Image(systemName: "checkmark")
                                .foregroundColor(.blue)
                        }
                    }
                }
                .buttonStyle(.plain)
            }
            .navigationTitle("Select Language")
            .navigationBarItems(trailing: Button("Done") { dismiss() })
        }
    }
}

// Make LanguageInfo identifiable for SwiftUI
extension LanguageInfo: Identifiable {
    public var id: String { code }
}