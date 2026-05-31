// react-native/UniversalTranslationSDK/android/src/main/java/com/universaltranslation/UniversalTranslationModule.kt

package com.universaltranslation

import android.util.Base64
import com.facebook.react.bridge.*
import com.facebook.react.modules.core.DeviceEventManagerModule
import com.universaltranslation.encoder.TranslationClient
import com.universaltranslation.encoder.TranslationResult
import com.universaltranslation.encoder.VocabularyManager
import com.universaltranslation.encoder.VocabularyPackManager
import kotlinx.coroutines.*
import java.util.concurrent.ConcurrentHashMap

class UniversalTranslationModule(reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext), CoroutineScope {

    override val coroutineContext = SupervisorJob() + Dispatchers.Main
    
    private var translationClient: TranslationClient? = null
    private var vocabularyManager: VocabularyManager? = null
    private var vocabularyPackManager: VocabularyPackManager? = null
    private val preparationJobs = ConcurrentHashMap<String, Job>()
    
    companion object {
        const val NAME = "UniversalTranslationModule"
        private const val TAG = "UniversalTranslation"
        
        // Supported languages mapping
        private val SUPPORTED_LANGUAGES = listOf(
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
            mapOf("code" to "pt", "name" to "Portuguese", "nativeName" to "Português", "isRTL" to false),
            mapOf("code" to "it", "name" to "Italian", "nativeName" to "Italiano", "isRTL" to false),
            mapOf("code" to "tr", "name" to "Turkish", "nativeName" to "Türkçe", "isRTL" to false),
            mapOf("code" to "th", "name" to "Thai", "nativeName" to "ไทย", "isRTL" to false),
            mapOf("code" to "vi", "name" to "Vietnamese", "nativeName" to "Tiếng Việt", "isRTL" to false),
            mapOf("code" to "pl", "name" to "Polish", "nativeName" to "Polski", "isRTL" to false),
            mapOf("code" to "uk", "name" to "Ukrainian", "nativeName" to "Українська", "isRTL" to false),
            mapOf("code" to "nl", "name" to "Dutch", "nativeName" to "Nederlands", "isRTL" to false),
            mapOf("code" to "id", "name" to "Indonesian", "nativeName" to "Bahasa Indonesia", "isRTL" to false),
            mapOf("code" to "sv", "name" to "Swedish", "nativeName" to "Svenska", "isRTL" to false)
        )
    }

    override fun getName(): String = NAME

    override fun onCatalystInstanceDestroy() {
        super.onCatalystInstanceDestroy()
        coroutineContext.cancelChildren()
        translationClient?.destroy()
    }

    @ReactMethod
    fun initialize(decoderUrl: String, promise: Promise) {
        launch {
            try {
                // Initialize translation client
                translationClient = TranslationClient(
                    context = reactApplicationContext,
                    decoderUrl = decoderUrl
                )
                
                // Initialize vocabulary managers
                vocabularyManager = VocabularyManager(reactApplicationContext)
                vocabularyPackManager = VocabularyPackManager(reactApplicationContext)
                
                promise.resolve(null)
            } catch (e: Exception) {
                promise.reject("INIT_ERROR", "Failed to initialize: ${e.message}", e)
            }
        }
    }

    @ReactMethod
    fun translate(text: String, sourceLang: String, targetLang: String, promise: Promise) {
        launch {
            try {
                val client = translationClient 
                    ?: throw IllegalStateException("Translation client not initialized")
                
                when (val result = client.translate(text, sourceLang, targetLang)) {
                    is TranslationResult.Success -> {
                        val resultMap = WritableNativeMap().apply {
                            putString("translation", result.translation)
                            putString("targetLang", targetLang)
                            putDouble("confidence", 1.0) // Add confidence if available
                        }
                        promise.resolve(resultMap)
                    }
                    is TranslationResult.Error -> {
                        promise.reject("TRANSLATION_ERROR", result.message)
                    }
                }
            } catch (e: Exception) {
                promise.reject("TRANSLATION_ERROR", e.message ?: "Translation failed", e)
            }
        }
    }

    @ReactMethod
    fun prepareTranslation(sourceLang: String, targetLang: String, promise: Promise) {
        launch {
            try {
                val client = translationClient?.encoder
                    ?: throw IllegalStateException("Translation client not initialized")
                
                // Cancel any existing preparation for this pair
                val key = "$sourceLang-$targetLang"
                preparationJobs[key]?.cancel()
                
                // Start new preparation job
                val job = launch {
                    val success = client.prepareTranslation(sourceLang, targetLang)
                    if (!success) {
                        throw Exception("Failed to prepare translation")
                    }
                }
                
                preparationJobs[key] = job
                job.join()
                preparationJobs.remove(key)
                
                promise.resolve(null)
            } catch (e: CancellationException) {
                promise.reject("PREPARE_CANCELLED", "Preparation was cancelled")
            } catch (e: Exception) {
                promise.reject("PREPARE_ERROR", e.message ?: "Preparation failed", e)
            }
        }
    }

    @ReactMethod
    fun getVocabularyForPair(sourceLang: String, targetLang: String, promise: Promise) {
        launch {
            try {
                val manager = vocabularyManager 
                    ?: throw IllegalStateException("Vocabulary manager not initialized")
                
                val pack = manager.getVocabularyForPair(sourceLang, targetLang)
                
                val packMap = WritableNativeMap().apply {
                    putString("name", pack.name)
                    putArray("languages", WritableNativeArray().apply {
                        pack.languages.forEach { pushString(it) }
                    })
                    putString("downloadUrl", pack.downloadUrl)
                    putString("localPath", pack.localPath)
                    putDouble("sizeMb", pack.sizeMb.toDouble())
                    putString("version", pack.version)
                    putBoolean("needsDownload", pack.needsDownload())
                }
                
                promise.resolve(packMap)
            } catch (e: Exception) {
                promise.reject("VOCAB_ERROR", e.message ?: "Failed to get vocabulary", e)
            }
        }
    }

    @ReactMethod
    fun downloadVocabularyPacks(languages: ReadableArray, promise: Promise) {
        launch {
            try {
                val packManager = vocabularyPackManager 
                    ?: throw IllegalStateException("Pack manager not initialized")
                
                val langSet = mutableSetOf<String>()
                for (i in 0 until languages.size()) {
                    langSet.add(languages.getString(i))
                }
                
                packManager.downloadPacksForLanguages(langSet)
                promise.resolve(null)
            } catch (e: Exception) {
                promise.reject("DOWNLOAD_ERROR", e.message ?: "Download failed", e)
            }
        }
    }

    @ReactMethod
    fun getSupportedLanguages(promise: Promise) {
        try {
            val languages = WritableNativeArray()
            
            SUPPORTED_LANGUAGES.forEach { lang ->
                languages.pushMap(WritableNativeMap().apply {
                    putString("code", lang["code"] as String)
                    putString("name", lang["name"] as String)
                    putString("nativeName", lang["nativeName"] as String)
                    putBoolean("isRTL", lang["isRTL"] as Boolean)
                })
            }
            
            promise.resolve(languages)
        } catch (e: Exception) {
            promise.reject("LANG_ERROR", e.message ?: "Failed to get languages", e)
        }
    }

    @ReactMethod
    fun getMemoryUsage(promise: Promise) {
        launch {
            try {
                val encoder = translationClient?.encoder
                    ?: throw IllegalStateException("Encoder not initialized")
                
                val memoryUsage = encoder.getMemoryUsage()
                promise.resolve(memoryUsage.toDouble())
            } catch (e: Exception) {
                promise.reject("MEMORY_ERROR", e.message ?: "Failed to get memory usage", e)
            }
        }
    }

    @ReactMethod
    fun clearTranslationCache(promise: Promise) {
        try {
            // TranslationClient doesn't have a clearCache method in the native implementation
            // This is a no-op for now, but we resolve successfully
            promise.resolve(null)
        } catch (e: Exception) {
            promise.reject("CACHE_ERROR", e.message ?: "Failed to clear cache", e)
        }
    }

    private fun sendEvent(eventName: String, params: WritableMap) {
        reactApplicationContext
            .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
            .emit(eventName, params)
    }
}