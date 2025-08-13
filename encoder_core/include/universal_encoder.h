// encoder_core/include/universal_encoder.h
#ifndef UNIVERSAL_ENCODER_H
#define UNIVERSAL_ENCODER_H

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <cstdint>  // Add for int32_t

// Forward declaration to avoid including heavy header if possible
namespace Ort {
    class Session;
    class MemoryInfo;
    class AllocatorWithDefaultOptions;
}

#include "onnxruntime_cxx_api.h"

namespace UniversalTranslation {

/**
 * @brief Vocabulary pack containing tokens and subwords for specific languages
 */
class VocabularyPack {
public:
    std::unordered_map<std::string, int32_t> tokens;
    std::unordered_map<std::string, int32_t> subwords;
    std::unordered_map<std::string, int32_t> special_tokens;
    std::vector<std::string> languages;
    std::string name;
    
    // Constructor/Destructor
    VocabularyPack() = default;
    ~VocabularyPack() = default;
    
    // Disable copy, enable move
    VocabularyPack(const VocabularyPack&) = delete;
    VocabularyPack& operator=(const VocabularyPack&) = delete;
    VocabularyPack(VocabularyPack&&) = default;
    VocabularyPack& operator=(VocabularyPack&&) = default;
    
    int32_t getTokenId(const std::string& token) const;
    std::vector<int32_t> tokenizeUnknown(const std::string& word) const;
    static std::unique_ptr<VocabularyPack> loadFromFile(const std::string& path);
};

/**
 * @brief Universal encoder for multilingual text encoding
 */
class UniversalEncoder {
private:
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::MemoryInfo> memory_info;
    std::unique_ptr<VocabularyPack> current_vocab;
    Ort::AllocatorWithDefaultOptions allocator;
    
    std::vector<int32_t> tokenize(const std::string& text, const std::string& source_lang);
    std::vector<float> runInference(const std::vector<int32_t>& tokens);
    
public:
    explicit UniversalEncoder(const std::string& model_path);
    ~UniversalEncoder();
    
    // Disable copy, enable move
    UniversalEncoder(const UniversalEncoder&) = delete;
    UniversalEncoder& operator=(const UniversalEncoder&) = delete;
    UniversalEncoder(UniversalEncoder&&) = default;
    UniversalEncoder& operator=(UniversalEncoder&&) = default;
    
    // Load vocabulary pack for language pair
    bool loadVocabulary(const std::string& vocab_path);
    
    // Encode text to embeddings
    std::vector<uint8_t> encode(
        const std::string& text,
        const std::string& source_lang,
        const std::string& target_lang
    );
    
    // Get supported languages
    std::vector<std::string> getSupportedLanguages() const;
    
    // Get current memory usage
    size_t getMemoryUsage() const;
};

} // namespace UniversalTranslation

#endif // UNIVERSAL_ENCODER_H