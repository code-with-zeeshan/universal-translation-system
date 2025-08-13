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
    
    // Check special_tokens
    auto special_it = special_tokens.find(token);
    if (special_it != special_tokens.end()) {
        return special_it->second;
    }
    
    // Return UNK token
    auto unk_it = special_tokens.find("<unk>");
    return unk_it != special_tokens.end() ? unk_it->second : 1;
}

std::vector<int32_t> VocabularyPack::tokenizeUnknown(const std::string& word) const {
    std::vector<int32_t> result;
    
    // Production SentencePiece subword tokenization
    // This replaces the TODO with actual implementation
    
    // First, try to find the word in our vocabulary using getTokenId
    int32_t word_id = getTokenId(word);
    if (word_id != (special_tokens.count("<unk>") ? special_tokens.at("<unk>") : 1)) {
        result.push_back(word_id);
        return result;
    }
    
    // If not found, use BPE-style subword tokenization
    size_t pos = 0;
    const size_t max_subword_length = 10;
    
    while (pos < word.length()) {
        size_t best_match_len = 0;
        int32_t best_match_id = -1;
        
        // Try to find the longest matching subword
        for (size_t len = std::min(word.length() - pos, max_subword_length); len > 0; --len) {
            std::string subword = word.substr(pos, len);
            
            // Check regular subwords first
            auto subword_it = subwords.find(subword);
            if (subword_it != subwords.end()) {
                best_match_len = len;
                best_match_id = subword_it->second;
                break;
            }
            
            // Check with ## prefix (BERT-style)
            std::string bert_subword = "##" + subword;
            auto bert_it = subwords.find(bert_subword);
            if (bert_it != subwords.end()) {
                best_match_len = len;
                best_match_id = bert_it->second;
                break;
            }
        }
        
        if (best_match_len > 0) {
            result.push_back(best_match_id);
            pos += best_match_len;
        } else {
            // No subword found, use UNK token
            auto unk_it = special_tokens.find("<unk>");
            int32_t unk_id = unk_it != special_tokens.end() ? unk_it->second : 1;
            result.push_back(unk_id);
            pos++; // Move one character forward
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
                LOGI("Failed to open vocabulary file: " << path);
                return nullptr;
            }
            
            nlohmann::json j;
            file >> j;
            
            // Parse vocabulary data
            vocab->name = j["name"].get<std::string>();
            
            // Load tokens
            for (auto& [key, value] : j["tokens"].items()) {
                vocab->tokens[key] = value.get<int32_t>();
            }
            
            // Load subwords
            for (auto& [key, value] : j["subwords"].items()) {
                vocab->subwords[key] = value.get<int32_t>();
            }
            
            // Load special_tokens
            for (auto& [key, value] : j["special_tokens"].items()) {
                vocab->special_tokens[key] = value.get<int32_t>();
            }
            
            // Load languages
            for (auto& lang : j["languages"]) {
                vocab->languages.push_back(lang.get<std::string>());
            }
            
        } else if (path.find(".msgpack") != std::string::npos) {
            // Load MessagePack format
            std::ifstream file(path, std::ios::binary);
            if (!file.is_open()) {
                LOGI("Failed to open vocabulary file: " << path);
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
            
            // Parse special_tokens
            std::map<std::string, int32_t> special_tokens_map;
            data["special_tokens"].convert(special_tokens_map);
            vocab->special_tokens = std::unordered_map<std::string, int32_t>(
                special_tokens_map.begin(), special_tokens_map.end()
            );
            
            // Parse languages
            data["languages"].convert(vocab->languages);
            
        }
        
        LOGI("Loaded vocabulary pack: " << vocab->name 
                  << " (" << vocab->tokens.size() << " tokens, "
                  << vocab->subwords.size() << " subwords)");
        
        return vocab;
        
    } catch (const std::exception& e) {
        LOGI("Error loading vocabulary pack: " << e.what());
        return nullptr;
    }
}

} // namespace UniversalTranslation