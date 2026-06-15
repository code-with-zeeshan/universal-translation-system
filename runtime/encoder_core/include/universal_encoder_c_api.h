#ifndef UNIVERSAL_ENCODER_C_API_H
#define UNIVERSAL_ENCODER_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void* universal_encoder_init(const char* model_path);
void universal_encoder_destroy(void* encoder);

int universal_encoder_load_vocabulary(void* encoder, const char* vocab_path);

uint8_t* universal_encoder_encode(
    void* encoder,
    const char* text,
    const char* source_lang,
    const char* target_lang,
    uint64_t* out_len
);

void universal_encoder_free_buffer(void* buffer);

char* universal_encoder_get_supported_languages(void* encoder);
void universal_encoder_free_string(char* str);

uint64_t universal_encoder_get_memory_usage(void* encoder);

#ifdef __cplusplus
}
#endif

#endif
