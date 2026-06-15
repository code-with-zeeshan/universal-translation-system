#include "universal_encoder_c_api.h"
#include "universal_encoder.h"
#include <cstring>
#include <string>

using namespace UniversalTranslation;

void* universal_encoder_init(const char* model_path) {
    try {
        auto* encoder = new UniversalEncoder(model_path);
        return static_cast<void*>(encoder);
    } catch (...) {
        return nullptr;
    }
}

void universal_encoder_destroy(void* encoder) {
    delete static_cast<UniversalEncoder*>(encoder);
}

int universal_encoder_load_vocabulary(void* encoder, const char* vocab_path) {
    if (!encoder || !vocab_path) return 0;
    auto* enc = static_cast<UniversalEncoder*>(encoder);
    return enc->loadVocabulary(vocab_path) ? 1 : 0;
}

uint8_t* universal_encoder_encode(
    void* encoder,
    const char* text,
    const char* source_lang,
    const char* target_lang,
    uint64_t* out_len
) {
    if (!encoder || !text || !source_lang || !target_lang || !out_len) {
        *out_len = 0;
        return nullptr;
    }
    try {
        auto* enc = static_cast<UniversalEncoder*>(encoder);
        auto result = enc->encode(text, source_lang, target_lang);
        auto* buf = static_cast<uint8_t*>(malloc(result.size()));
        if (!buf) {
            *out_len = 0;
            return nullptr;
        }
        memcpy(buf, result.data(), result.size());
        *out_len = result.size();
        return buf;
    } catch (...) {
        *out_len = 0;
        return nullptr;
    }
}

void universal_encoder_free_buffer(void* buffer) {
    free(buffer);
}

char* universal_encoder_get_supported_languages(void* encoder) {
    if (!encoder) return nullptr;
    auto* enc = static_cast<UniversalEncoder*>(encoder);
    auto langs = enc->getSupportedLanguages();
    std::string joined;
    for (size_t i = 0; i < langs.size(); ++i) {
        if (i > 0) joined += ",";
        joined += langs[i];
    }
    auto* result = static_cast<char*>(malloc(joined.size() + 1));
    if (!result) return nullptr;
    memcpy(result, joined.data(), joined.size());
    result[joined.size()] = '\0';
    return result;
}

void universal_encoder_free_string(char* str) {
    free(str);
}

uint64_t universal_encoder_get_memory_usage(void* encoder) {
    if (!encoder) return 0;
    auto* enc = static_cast<UniversalEncoder*>(encoder);
    return static_cast<uint64_t>(enc->getMemoryUsage());
}
