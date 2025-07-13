// encoder_core/src/universal_encoder.cpp
#include "universal_encoder.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <lz4.h>
#include "nlohmann/json.hpp"

namespace UniversalTranslation {

UniversalEncoder::UniversalEncoder(const std::string& model_path) {
    // Initialize ONNX Runtime
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(2);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    // Use NNAPI on Android, CoreML on iOS
    #ifdef __ANDROID__
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(session_options));
    #elif __APPLE__
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(session_options));
    #endif
    
    session = std::make_unique<Ort::Session>(Ort::Env{ORT_LOGGING_LEVEL_WARNING, "encoder"}, 
                                             model_path.c_str(), session_options);
    
    memory_info = std::make_unique<Ort::MemoryInfo>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
    );
}

bool UniversalEncoder::loadVocabulary(const std::string& vocab_path) {
    current_vocab = VocabularyPack::loadFromFile(vocab_path);
    return current_vocab != nullptr;
}

std::vector<int32_t> UniversalEncoder::tokenize(const std::string& text, const std::string& source_lang) {
    std::vector<int32_t> tokens;
    
    // Add language token
    std::string lang_token = "<" + source_lang + ">";
    auto it = current_vocab->tokens.find(lang_token);
    if (it != current_vocab->tokens.end()) {
        tokens.push_back(it->second);
    }
    
    // Simple whitespace tokenization (production would use SentencePiece)
    std::istringstream iss(text);
    std::string word;
    
    while (iss >> word) {
        // Convert to lowercase
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
        
        // Look up token
        auto token_it = current_vocab->tokens.find(word);
        if (token_it != current_vocab->tokens.end()) {
            tokens.push_back(token_it->second);
        } else {
            // Handle unknown word with subword tokenization
            auto subwords = current_vocab->tokenizeUnknown(word);
            tokens.insert(tokens.end(), subwords.begin(), subwords.end());
        }
    }
    
    // Add end token
    tokens.push_back(current_vocab->tokens.at("</s>"));
    
    // Pad to 128
    while (tokens.size() < 128) {
        tokens.push_back(current_vocab->tokens.at("<pad>"));
    }
    
    if (tokens.size() > 128) {
        tokens.resize(128);
    }
    
    return tokens;
}

std::vector<uint8_t> UniversalEncoder::encode(
    const std::string& text,
    const std::string& source_lang,
    const std::string& target_lang
) {
    // Tokenize
    auto tokens = tokenize(text, source_lang);
    
    // Prepare input tensor
    std::vector<int64_t> input_shape = {1, 128};
    auto input_tensor = Ort::Value::CreateTensor<int32_t>(
        *memory_info, tokens.data(), tokens.size(), 
        input_shape.data(), input_shape.size()
    );
    
    // Run inference
    auto output_tensors = session->Run(
        Ort::RunOptions{nullptr},
        {"input_ids"}, &input_tensor, 1,
        {"encoder_output"}, 1
    );
    
    // Get output
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t output_size = output_shape[1] * output_shape[2];
    
    // Quantize to int8 for compression
    std::vector<int8_t> quantized(output_size);
    float scale = 127.0f;
    
    for (size_t i = 0; i < output_size; ++i) {
        float max_val = 0;
        for (size_t j = 0; j < output_size; ++j) {
            max_val = std::max(max_val, std::abs(output_data[j]));
        }
        scale = max_val > 0 ? 127.0f / max_val : 1.0f;
        break;
    }
    
    for (size_t i = 0; i < output_size; ++i) {
        quantized[i] = static_cast<int8_t>(std::round(output_data[i] * scale));
    }
    
    // Compress with LZ4
    int compressed_size = LZ4_compressBound(quantized.size());
    std::vector<uint8_t> compressed(compressed_size + 12); // +12 for metadata
    
    // Add metadata (shape, scale, etc.)
    *reinterpret_cast<int32_t*>(&compressed[0]) = output_shape[1];
    *reinterpret_cast<int32_t*>(&compressed[4]) = output_shape[2];
    *reinterpret_cast<float*>(&compressed[8]) = scale;
    
    // Compress
    int actual_size = LZ4_compress_default(
        reinterpret_cast<const char*>(quantized.data()),
        reinterpret_cast<char*>(&compressed[12]),
        quantized.size(),
        compressed_size
    );
    
    compressed.resize(12 + actual_size);
    return compressed;
}

} // namespace