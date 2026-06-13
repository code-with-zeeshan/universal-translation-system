// encoder_core/examples/example_usage.cpp
#include "universal_encoder.h"
#include <iostream>
#include <chrono>

using namespace UniversalTranslation;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <vocab_path>" << std::endl;
        return 1;
    }
    
    try {
        // Initialize encoder
        std::cout << "Loading model..." << std::endl;
        UniversalEncoder encoder(argv[1]);
        
        // Load vocabulary
        std::cout << "Loading vocabulary..." << std::endl;
        if (!encoder.loadVocabulary(argv[2])) {
            std::cerr << "Failed to load vocabulary" << std::endl;
            return 1;
        }
        
        // Get supported languages
        auto languages = encoder.getSupportedLanguages();
        std::cout << "Supported languages: ";
        for (const auto& lang : languages) {
            std::cout << lang << " ";
        }
        std::cout << std::endl;
        
        // Test encoding
        std::string test_text = "Hello, how are you today?";
        std::string source_lang = "en";
        std::string target_lang = "es";
        
        std::cout << "\nEncoding: \"" << test_text << "\"" << std::endl;
        std::cout << "Source: " << source_lang << ", Target: " << target_lang << std::endl;
        
        // Measure encoding time
        auto start = std::chrono::high_resolution_clock::now();
        
        auto encoded = encoder.encode(test_text, source_lang, target_lang);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Encoded size: " << encoded.size() << " bytes" << std::endl;
        std::cout << "Encoding time: " << duration.count() << " ms" << std::endl;
        std::cout << "Memory usage: " << encoder.getMemoryUsage() / (1024*1024) << " MB" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}