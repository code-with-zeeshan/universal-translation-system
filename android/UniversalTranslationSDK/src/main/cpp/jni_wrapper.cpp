// android/UniversalTranslationSDK/src/main/cpp/jni_wrapper.cpp

#include <jni.h>
#include <string>
#include <android/log.h>
#include <memory>
#include "universal_encoder.h"

#define LOG_TAG "UniversalEncoderJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using namespace UniversalTranslation;

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_universaltranslation_encoder_TranslationEncoder_nativeInit(
    JNIEnv *env, jclass clazz, jstring model_path) {
    
    const char *path = env->GetStringUTFChars(model_path, nullptr);
    LOGI("Initializing encoder with model: %s", path);
    
    try {
        auto encoder = std::make_unique<UniversalEncoder>(path);
        env->ReleaseStringUTFChars(model_path, path);
        
        // Return pointer as long
        return reinterpret_cast<jlong>(encoder.release());
    } catch (const std::exception& e) {
        LOGE("Failed to initialize encoder: %s", e.what());
        env->ReleaseStringUTFChars(model_path, path);
        return 0;
    }
}

JNIEXPORT jboolean JNICALL
Java_com_universaltranslation_encoder_TranslationEncoder_nativeLoadVocabulary(
    JNIEnv *env, jclass clazz, jlong handle, jstring vocab_path) {
    
    if (handle == 0) return JNI_FALSE;
    
    auto encoder = reinterpret_cast<UniversalEncoder*>(handle);
    const char *path = env->GetStringUTFChars(vocab_path, nullptr);
    
    try {
        bool result = encoder->loadVocabulary(path);
        env->ReleaseStringUTFChars(vocab_path, path);
        return result ? JNI_TRUE : JNI_FALSE;
    } catch (const std::exception& e) {
        LOGE("Failed to load vocabulary: %s", e.what());
        env->ReleaseStringUTFChars(vocab_path, path);
        return JNI_FALSE;
    }
}

JNIEXPORT jbyteArray JNICALL
Java_com_universaltranslation_encoder_TranslationEncoder_nativeEncode(
    JNIEnv *env, jclass clazz, jlong handle, 
    jstring text, jstring source_lang, jstring target_lang) {
    
    if (handle == 0) {
        LOGE("Invalid encoder handle");
        return nullptr;
    }
    
    auto encoder = reinterpret_cast<UniversalEncoder*>(handle);
    
    // Get strings
    const char *text_str = env->GetStringUTFChars(text, nullptr);
    const char *source_str = env->GetStringUTFChars(source_lang, nullptr);
    const char *target_str = env->GetStringUTFChars(target_lang, nullptr);
    
    try {
        // Encode
        auto encoded = encoder->encode(text_str, source_str, target_str);
        
        // Create byte array
        jbyteArray result = env->NewByteArray(encoded.size());
        env->SetByteArrayRegion(result, 0, encoded.size(), 
                               reinterpret_cast<const jbyte*>(encoded.data()));
        
        // Release strings
        env->ReleaseStringUTFChars(text, text_str);
        env->ReleaseStringUTFChars(source_lang, source_str);
        env->ReleaseStringUTFChars(target_lang, target_str);
        
        return result;
    } catch (const std::exception& e) {
        LOGE("Encoding failed: %s", e.what());
        
        // Release strings
        env->ReleaseStringUTFChars(text, text_str);
        env->ReleaseStringUTFChars(source_lang, source_str);
        env->ReleaseStringUTFChars(target_lang, target_str);
        
        return nullptr;
    }
}

JNIEXPORT void JNICALL
Java_com_universaltranslation_encoder_TranslationEncoder_nativeDestroy(
    JNIEnv *env, jclass clazz, jlong handle) {
    
    if (handle != 0) {
        auto encoder = reinterpret_cast<UniversalEncoder*>(handle);
        delete encoder;
        LOGI("Encoder destroyed");
    }
}

JNIEXPORT jlong JNICALL
Java_com_universaltranslation_encoder_TranslationEncoder_nativeGetMemoryUsage(
    JNIEnv *env, jclass clazz, jlong handle) {
    
    if (handle == 0) return 0;
    
    auto encoder = reinterpret_cast<UniversalEncoder*>(handle);
    return static_cast<jlong>(encoder->getMemoryUsage());
}

} // extern "C"