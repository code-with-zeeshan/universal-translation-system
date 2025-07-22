// encoder_core/src/vocabulary_pack.cpp
#include "universal_encoder.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <msgpack.hpp>
#include "nlohmann/json.hpp"

namespace UniversalTranslation {

int32_t VocabularyPack::getTokenId(const std::string& token) const {
    auto it = tokens.find(token);
    if (it != tokens.end()) {
        return it->second;
    }
    
    // Check subwords
    auto subword_it = subwords.find(token);
    if (subword_it != subwords.end()) {
        return subword_it->second;
    }
    
    // Return UNK token
    auto unk_it = tokens.find("<unk>");
    return unk_it != tokens.end() ? unk_it->second : 1;
}

std::vector<int32_t> VocabularyPack::tokenizeUnknown(const std::string& word) const {
    std::vector<int32_t> result;
    // TODO: Integrate production BPE or SentencePiece subword tokenization here
    // For now, fallback to simple subword logic
    size_t pos = 0;
    while (pos < word.length()) {
        size_t best_match_len = 0;
        int32_t best_match_id = -1;
        for (size_t len = std::min(word.length() - pos, size_t(10)); len > 0; --len) {
            std::string subword = "##" + word.substr(pos, len);
            auto it = subwords.find(subword);
            if (it != subwords.end()) {
                best_match_len = len;
                best_match_id = it->second;
                break;
            }
        }
        if (best_match_len > 0) {
            result.push_back(best_match_id);
            pos += best_match_len;
        } else {
            result.push_back(getTokenId("<unk>"));
            pos++;
        }
    }
    return result;
}

std::unique_ptr<VocabularyPack> VocabularyPack::loadFromFile(const std::string& path) {
    auto vocab = std::make_unique<VocabularyPack>();
    
    try {
        // Determine file format by extension
        if (path.find(".json") != std::string::npos) {
            // Load JSON format
            std::ifstream file(path);
            if (!file.is_open()) {
                std::cerr << "Failed to open vocabulary file: " << path << std::endl;
                return nullptr;
            }
            
            nlohmann::json j;
            file >> j;
            
            // Parse vocabulary data
            vocab->name = j["name"].get<std::string>();
            vocab->size_mb = j["metadata"]["size_mb"].get<float>();
            
            // Load tokens
            for (auto& [key, value] : j["tokens"].items()) {
                vocab->tokens[key] = value.get<int32_t>();
            }
            
            // Load subwords
            for (auto& [key, value] : j["subwords"].items()) {
                vocab->subwords[key] = value.get<int32_t>();
            }
            
            // Load languages
            for (auto& lang : j["languages"]) {
                vocab->languages.push_back(lang.get<std::string>());
            }
            
        } else if (path.find(".msgpack") != std::string::npos) {
            // Load MessagePack format
            std::ifstream file(path, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Failed to open vocabulary file: " << path << std::endl;
                return nullptr;
            }
            
            // Read file contents
            std::string contents((std::istreambuf_iterator<char>(file)),
                                std::istreambuf_iterator<char>());
            
            // Unpack MessagePack data
            msgpack::object_handle oh = msgpack::unpack(contents.data(), contents.size());
            msgpack::object obj = oh.get();
            
            // Convert to map
            std::map<std::string, msgpack::object> data;
            obj.convert(data);
            
            // Parse data
            data["name"].convert(vocab->name);
            
            // Parse tokens
            std::map<std::string, int32_t> tokens_map;
            data["tokens"].convert(tokens_map);
            vocab->tokens = std::unordered_map<std::string, int32_t>(
                tokens_map.begin(), tokens_map.end()
            );
            
            // Parse subwords
            std::map<std::string, int32_t> subwords_map;
            data["subwords"].convert(subwords_map);
            vocab->subwords = std::unordered_map<std::string, int32_t>(
                subwords_map.begin(), subwords_map.end()
            );
            
            // Parse languages
            data["languages"].convert(vocab->languages);
            
            // Parse metadata
            std::map<std::string, msgpack::object> metadata;
            data["metadata"].convert(metadata);
            metadata["size_mb"].convert(vocab->size_mb);
        }
        
        std::cout << "Loaded vocabulary pack: " << vocab->name 
                  << " (" << vocab->tokens.size() << " tokens, "
                  << vocab->subwords.size() << " subwords)" << std::endl;
        
        return vocab;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading vocabulary pack: " << e.what() << std::endl;
        return nullptr;
    }
}

} // namespace UniversalTranslation