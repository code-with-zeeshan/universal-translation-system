// encoder_core/src/universal_encoder.cpp
#include "universal_encoder.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <lz4.h>
#include "nlohmann/json.hpp"

namespace UniversalTranslation {

UniversalEncoder::UniversalEncoder(const std::string& model_path) {
    try {
        // Initialize ONNX Runtime environment
        static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "UniversalEncoder");
        
        // Configure session options
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(2);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // Platform-specific execution providers
        #ifdef __ANDROID__
            // Use NNAPI on Android for hardware acceleration
            uint32_t nnapi_flags = 0;
            nnapi_flags |= NNAPI_FLAG_USE_FP16;
            Ort::ThrowOnError(
                OrtSessionOptionsAppendExecutionProvider_Nnapi(session_options, nnapi_flags)
            );
        #elif defined(__APPLE__)
            // Use CoreML on iOS/macOS
            uint32_t coreml_flags = 0;
            coreml_flags |= COREML_FLAG_ENABLE_ON_SUBGRAPH;
            Ort::ThrowOnError(
                OrtSessionOptionsAppendExecutionProvider_CoreML(session_options, coreml_flags)
            );
        #endif

        // Add proper platform detection
        #ifdef __ANDROID__
            #include <android/log.h>
            #define LOG_TAG "UniversalEncoder"
            #define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
        #else
            #define LOGI(...) std::cout << __VA_ARGS__ << std::endl
        #endif
        
        // Create session
        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
        
        // Create memory info
        memory_info = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
        );
        
        // Log model info
        size_t num_input_nodes = session->GetInputCount();
        size_t num_output_nodes = session->GetOutputCount();
        
        std::cout << "Model loaded successfully:" << std::endl;
        std::cout << "  Input nodes: " << num_input_nodes << std::endl;
        std::cout << "  Output nodes: " << num_output_nodes << std::endl;
        
        // Get input/output names and shapes
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = session->GetInputNameAllocated(i, allocator);
            auto type_info = session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            
            std::cout << "  Input " << i << ": " << input_name.get() 
                      << " shape: [";
            for (auto dim : tensor_info.GetShape()) {
                std::cout << dim << " ";
            }
            std::cout << "]" << std::endl;
        }
        
    } catch (const Ort::Exception& e) {
        throw std::runtime_error("Failed to initialize ONNX Runtime: " + std::string(e.what()));
    }
}

UniversalEncoder::~UniversalEncoder() {
    // Cleanup is handled by smart pointers
}

bool UniversalEncoder::loadVocabulary(const std::string& vocab_path) {
    try {
        current_vocab = VocabularyPack::loadFromFile(vocab_path);
        return current_vocab != nullptr;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load vocabulary: " << e.what() << std::endl;
        return false;
    }
}

std::vector<int32_t> UniversalEncoder::tokenize(const std::string& text, const std::string& source_lang) {
    if (!current_vocab) {
        throw std::runtime_error("No vocabulary loaded");
    }
    std::vector<int32_t> tokens;
    // Add BOS token
    tokens.push_back(current_vocab->getTokenId("<s>"));
    // Add language token
    std::string lang_token = "<" + source_lang + ">";
    tokens.push_back(current_vocab->getTokenId(lang_token));
    // Production: Use SentencePiece or similar tokenizer
    // TODO: Integrate SentencePiece tokenizer here
    // Example placeholder:
    // std::vector<std::string> sp_tokens = sentencepiece_tokenizer.Encode(text);
    // for (const auto& tok : sp_tokens) {
    //     tokens.push_back(current_vocab->getTokenId(tok));
    // }
    // For now, fallback to whitespace tokenization
    std::istringstream iss(text);
    std::string word;
    while (iss >> word) {
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
        auto token_it = current_vocab->tokens.find(word);
        if (token_it != current_vocab->tokens.end()) {
            tokens.push_back(token_it->second);
        } else {
            auto subwords = current_vocab->tokenizeUnknown(word);
            tokens.insert(tokens.end(), subwords.begin(), subwords.end());
        }
    }
    // Add EOS token
    tokens.push_back(current_vocab->getTokenId("</s>"));
    // Pad or truncate to 128 tokens
    const size_t max_length = 128;
    if (tokens.size() < max_length) {
        int32_t pad_id = current_vocab->getTokenId("<pad>");
        tokens.resize(max_length, pad_id);
    } else if (tokens.size() > max_length) {
        tokens.resize(max_length - 1);
        tokens.push_back(current_vocab->getTokenId("</s>"));
    }
    return tokens;
}

std::vector<float> UniversalEncoder::runInference(const std::vector<int32_t>& tokens) {
    // Prepare input tensor
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(tokens.size())};
    
    // Create input tensor
    auto input_tensor = Ort::Value::CreateTensor<int32_t>(
        *memory_info, 
        const_cast<int32_t*>(tokens.data()), 
        tokens.size(),
        input_shape.data(), 
        input_shape.size()
    );
    
    // Get input/output names
    auto input_name = session->GetInputNameAllocated(0, allocator);
    auto output_name = session->GetOutputNameAllocated(0, allocator);
    
    const char* input_names[] = {input_name.get()};
    const char* output_names[] = {output_name.get()};
    
    // Run inference
    auto output_tensors = session->Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1
    );
    
    // Get output data
    auto& output_tensor = output_tensors[0];
    auto output_info = output_tensor.GetTensorTypeAndShapeInfo();
    auto output_shape = output_info.GetShape();
    
    // Calculate total size
    size_t total_size = std::accumulate(
        output_shape.begin(), 
        output_shape.end(), 
        size_t(1), 
        std::multiplies<size_t>()
    );
    
    // Get output data based on type
    std::vector<float> output_data;
    output_data.reserve(total_size);
    
    auto elem_type = output_info.GetElementType();
    if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        // Float32 output
        float* raw_output = output_tensor.GetTensorMutableData<float>();
        output_data.assign(raw_output, raw_output + total_size);
    } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
        // INT8 quantized output
        int8_t* raw_output = output_tensor.GetTensorMutableData<int8_t>();
        output_data.resize(total_size);
        
        // Dequantize (assuming scale of 127)
        float scale = 1.0f / 127.0f;
        for (size_t i = 0; i < total_size; ++i) {
            output_data[i] = static_cast<float>(raw_output[i]) * scale;
        }
    } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        // FP16 output - would need conversion
        throw std::runtime_error("FP16 output not yet supported");
    }
    
    return output_data;
}

std::vector<uint8_t> UniversalEncoder::encode(
    const std::string& text,
    const std::string& source_lang,
    const std::string& target_lang
) {
    // Tokenize input text
    auto tokens = tokenize(text, source_lang);
    
    // Run inference
    auto output_data = runInference(tokens);
    
    // Get output shape from session
    auto output_info = session->GetOutputTypeInfo(0);
    auto tensor_info = output_info.GetTensorTypeAndShapeInfo();
    auto output_shape = tensor_info.GetShape();
    
    // Quantize output to INT8 for compression
    std::vector<int8_t> quantized;
    quantized.reserve(output_data.size());
    
    // Find scale factor for quantization
    float max_abs_val = 0.0f;
    for (const auto& val : output_data) {
        max_abs_val = std::max(max_abs_val, std::abs(val));
    }
    
    float scale = max_abs_val > 0 ? 127.0f / max_abs_val : 1.0f;
    
    // Quantize
    for (const auto& val : output_data) {
        int quantized_val = static_cast<int>(std::round(val * scale));
        quantized_val = std::clamp(quantized_val, -128, 127);
        quantized.push_back(static_cast<int8_t>(quantized_val));
    }
    
    // Prepare compressed output
    int max_compressed_size = LZ4_compressBound(quantized.size());
    std::vector<uint8_t> compressed;
    compressed.reserve(max_compressed_size + 16); // Extra space for metadata
    
    // Add metadata header (16 bytes)
    // [0-3]: sequence length (int32)
    // [4-7]: hidden dimension (int32)
    // [8-11]: scale factor (float32)
    // [12-15]: reserved for future use
    
    compressed.resize(16);
    *reinterpret_cast<int32_t*>(&compressed[0]) = static_cast<int32_t>(output_shape[1]);
    *reinterpret_cast<int32_t*>(&compressed[4]) = static_cast<int32_t>(output_shape[2]);
    *reinterpret_cast<float*>(&compressed[8]) = scale;
    *reinterpret_cast<int32_t*>(&compressed[12]) = 0; // Reserved
    
    // Compress quantized data
    compressed.resize(16 + max_compressed_size);
    int compressed_size = LZ4_compress_default(
        reinterpret_cast<const char*>(quantized.data()),
        reinterpret_cast<char*>(&compressed[16]),
        quantized.size(),
        max_compressed_size
    );
    
    if (compressed_size <= 0) {
        throw std::runtime_error("Compression failed");
    }
    
    // Resize to actual size
    compressed.resize(16 + compressed_size);
    
    // Log compression stats
    float compression_ratio = static_cast<float>(quantized.size()) / compressed_size;
    std::cout << "Compression: " << quantized.size() << " bytes -> " 
              << compressed_size << " bytes (ratio: " << compression_ratio << "x)" << std::endl;
    
    return compressed;
}

std::vector<std::string> UniversalEncoder::getSupportedLanguages() const {
    if (!current_vocab) {
        return {};
    }
    return current_vocab->languages;
}

size_t UniversalEncoder::getMemoryUsage() const {
    size_t total_memory = 0;
    
    // Estimate vocabulary memory
    if (current_vocab) {
        total_memory += current_vocab->tokens.size() * (sizeof(std::string) + sizeof(int32_t) + 32);
        total_memory += current_vocab->subwords.size() * (sizeof(std::string) + sizeof(int32_t) + 32);
    }
    
    // Note: Getting actual ONNX Runtime memory usage is complex
    // This is a rough estimate
    total_memory += 100 * 1024 * 1024; // Assume ~100MB for model
    
    return total_memory;
}

} // namespace UniversalTranslation