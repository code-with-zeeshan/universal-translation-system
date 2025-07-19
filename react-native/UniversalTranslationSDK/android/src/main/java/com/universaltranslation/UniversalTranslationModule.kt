// react-native/UniversalTranslationSDK/android/src/main/java/com/universaltranslation/UniversalTranslationModule.kt

package com.universaltranslation

import android.util.Base64
import com.facebook.react.bridge.*
import com.facebook.react.modules.core.DeviceEventManagerModule
import com.universaltranslation.encoder.TranslationEncoder
import com.universaltranslation.encoder.TranslationResult
import com.universaltranslation.encoder.VocabularyPack
import kotlinx.coroutines.*
import java.util.concurrent.ConcurrentHashMap

class UniversalTranslationModule(reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext), CoroutineScope {

    override val coroutineContext = SupervisorJob() + Dispatchers.Main
    
    private var encoder: TranslationEncoder? = null
    private val downloadJobs = ConcurrentHashMap<String, Job>()
    
    companion object {
        const val NAME = "UniversalTranslationModule"
        private const val TAG = "UniversalTranslation"
    }

    override fun getName(): String = NAME

    override fun onCatalystInstanceDestroy() {
        super.onCatalystInstanceDestroy()
        coroutineContext.cancelChildren()
        encoder?.destroy()
    }

    @ReactMethod
    fun initialize(promise: Promise) {
        launch {
            try {
                if (encoder == null) {
                    encoder = TranslationEncoder(reactApplicationContext)
                }
                promise.resolve(null)
            } catch (e: Exception) {
                promise.reject("INIT_ERROR", "Failed to initialize encoder", e)
            }
        }
    }

    @ReactMethod
    fun prepareTranslation(sourceLang: String, targetLang: String, promise: Promise) {
        launch {
            try {
                val enc = encoder ?: throw IllegalStateException("Encoder not initialized")
                val success = enc.prepareTranslation(sourceLang, targetLang)
                
                if (success) {
                    promise.resolve(null)
                } else {
                    promise.reject("PREPARE_ERROR", "Failed to prepare translation")
                }
            } catch (e: Exception) {
                promise.reject("PREPARE_ERROR", e.message, e)
            }
        }
    }

    @ReactMethod
    fun encode(text: String, sourceLang: String, targetLang: String, promise: Promise) {
        launch {
            try {
                val enc = encoder ?: throw IllegalStateException("Encoder not initialized")
                val encoded = enc.encode(text, sourceLang, targetLang)
                
                // Convert to base64 for React Native bridge
                val base64 = Base64.encodeToString(encoded, Base64.NO_WRAP)
                promise.resolve(base64)
            } catch (e: Exception) {
                promise.reject("ENCODE_ERROR", e.message, e)
            }
        }
    }

    @ReactMethod
    fun getAvailableVocabularies(promise: Promise) {
        launch {
            try {
                val enc = encoder ?: throw IllegalStateException("Encoder not initialized")
                val vocabs = enc.getAvailableVocabularies()
                
                val vocabArray = WritableNativeArray()
                vocabs.forEach { vocab ->
                    val vocabMap = WritableNativeMap().apply {
                        putString("name", vocab.name)
                        putArray("languages", WritableNativeArray().apply {
                            vocab.languages.forEach { pushString(it) }
                        })
                        putDouble("sizeMB", vocab.sizeMb.toDouble())
                        putBoolean("isDownloaded", !vocab.needsDownload())
                        putString("version", vocab.version)
                    }
                    vocabArray.pushMap(vocabMap)
                }
                
                promise.resolve(vocabArray)
            } catch (e: Exception) {
                promise.reject("VOCAB_ERROR", e.message, e)
            }
        }
    }

    @ReactMethod
    fun downloadVocabulary(name: String, promise: Promise) {
        launch {
            try {
                val enc = encoder ?: throw IllegalStateException("Encoder not initialized")
                
                // Cancel any existing download for this vocabulary
                downloadJobs[name]?.cancel()
                
                // Start new download job
                val job = launch {
                    enc.downloadVocabulary(name) { progress ->
                        sendEvent("vocabularyDownloadProgress", WritableNativeMap().apply {
                            putString("name", name)
                            putDouble("progress", progress.toDouble())
                        })
                    }
                }
                
                downloadJobs[name] = job
                job.join()
                downloadJobs.remove(name)
                
                promise.resolve(null)
            } catch (e: CancellationException) {
                downloadJobs.remove(name)
                promise.reject("DOWNLOAD_CANCELLED", "Download was cancelled")
            } catch (e: Exception) {
                downloadJobs.remove(name)
                promise.reject("DOWNLOAD_ERROR", e.message, e)
            }
        }
    }

    @ReactMethod
    fun deleteVocabulary(name: String, promise: Promise) {
        launch {
            try {
                val enc = encoder ?: throw IllegalStateException("Encoder not initialized")
                enc.deleteVocabulary(name)
                promise.resolve(null)
            } catch (e: Exception) {
                promise.reject("DELETE_ERROR", e.message, e)
            }
        }
    }

    @ReactMethod
    fun getSupportedLanguages(promise: Promise) {
        launch {
            try {
                val languages = WritableNativeArray()
                
                val supportedLangs = listOf(
                    mapOf("code" to "en", "name" to "English", "nativeName" to "English", "isRTL" to false),
                    mapOf("code" to "es", "name" to "Spanish", "nativeName" to "Español", "isRTL" to false),
                    mapOf("code" to "fr", "name" to "French", "nativeName" to "Français", "isRTL" to false),
                    mapOf("code" to "de", "name" to "German", "nativeName" to "Deutsch", "isRTL" to false),
                    mapOf("code" to "zh", "name" to "Chinese", "nativeName" to "中文", "isRTL" to false),
                    mapOf("code" to "ja", "name" to "Japanese", "nativeName" to "日本語", "isRTL" to false),
                    mapOf("code" to "ko", "name" to "Korean", "nativeName" to "한국어", "isRTL" to false),
                    mapOf("code" to "ar", "name" to "Arabic", "nativeName" to "العربية", "isRTL" to true),
                    mapOf("code" to "hi", "name" to "Hindi", "nativeName" to "हिन्दी", "isRTL" to false),
                    mapOf("code" to "ru", "name" to "Russian", "nativeName" to "Русский", "isRTL" to false),
                    mapOf("code" to "pt", "name" to "Portuguese", "nativeName" to "Português", "isRTL" to false)
                )
                
                supportedLangs.forEach { lang ->
                    languages.pushMap(WritableNativeMap().apply {
                        putString("code", lang["code"] as String)
                        putString("name", lang["name"] as String)
                        putString("nativeName", lang["nativeName"] as String)
                        putBoolean("isRTL", lang["isRTL"] as Boolean)
                    })
                }
                
                promise.resolve(languages)
            } catch (e: Exception) {
                promise.reject("LANG_ERROR", e.message, e)
            }
        }
    }

    @ReactMethod
    fun getMemoryUsage(promise: Promise) {
        launch {
            try {
                val enc = encoder ?: throw IllegalStateException("Encoder not initialized")
                val memoryUsage = enc.getMemoryUsage()
                promise.resolve(memoryUsage.toDouble())
            } catch (e: Exception) {
                promise.reject("MEMORY_ERROR", e.message, e)
            }
        }
    }

    @ReactMethod
    fun clearCache(promise: Promise) {
        launch {
            try {
                val enc = encoder ?: throw IllegalStateException("Encoder not initialized")
                enc.clearCache()
                promise.resolve(null)
            } catch (e: Exception) {
                promise.reject("CACHE_ERROR", e.message, e)
            }
        }
    }

    private fun sendEvent(eventName: String, params: WritableMap) {
        reactApplicationContext
            .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
            .emit(eventName, params)
    }
}