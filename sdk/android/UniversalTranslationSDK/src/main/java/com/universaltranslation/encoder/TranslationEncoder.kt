// android/UniversalTranslationSDK/src/main/java/com/universaltranslation/encoder/TranslationEncoder.kt
package com.universaltranslation.encoder

import android.content.Context
import android.os.Build
import android.util.Log
import androidx.annotation.Keep
import com.google.gson.Gson
import kotlinx.coroutines.*
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.File
import java.io.IOException
import java.util.concurrent.TimeUnit

// Data classes
@Keep
data class PerformanceMetrics(
    val operation: String,
    val duration: Long,
    val timestamp: Long = System.currentTimeMillis()
)

@Keep
data class VocabularyPack(
    val name: String,
    val languages: List<String>,
    val downloadUrl: String,
    val localPath: String,
    val sizeMb: Float,
    val version: String,
    val tokens: Map<String, Int>? = null
) {
    fun needsDownload(): Boolean {
        return !File(localPath).exists()
    }
}

@Keep
data class TranslationResponse(
    val translation: String,
    val targetLang: String,
    val scores: List<Float>? = null
)

@Keep
sealed class TranslationResult {
    data class Success(val translation: String) : TranslationResult()
    data class Error(val message: String) : TranslationResult()
}

// Performance monitor
object PerformanceMonitor {
    private val metrics = mutableListOf<PerformanceMetrics>()
    
    inline fun <T> measureTime(operation: String, block: () -> T): T {
        val startTime = System.currentTimeMillis()
        return try {
            block()
        } finally {
            val duration = System.currentTimeMillis() - startTime
            metrics.add(PerformanceMetrics(operation, duration))
            
            if (duration > 1000) {
                Log.w("PerformanceMonitor", "$operation took ${duration}ms")
            }
        }
    }
    
    fun getMetrics(): List<PerformanceMetrics> = metrics.toList()
    fun clearMetrics() = metrics.clear()
}

// Vocabulary Manager
class VocabularyManager(private val context: Context) {
    
    companion object {
        const val TAG = "VocabularyManager"
        const val VOCAB_DIR = "vocabularies"
        val LANGUAGE_TO_PACK = mapOf(
            "en" to "latin", "es" to "latin", "fr" to "latin",
            "de" to "latin", "it" to "latin", "pt" to "latin",
            "zh" to "cjk", "ja" to "cjk", "ko" to "cjk",
            "ar" to "arabic", "hi" to "devanagari",
            "ru" to "cyrillic", "uk" to "cyrillic"
        )
    }
    
    private val vocabDir = File(context.filesDir, VOCAB_DIR).apply {
        if (!exists()) mkdirs()
    }
    
    fun getVocabularyForPair(sourceLang: String, targetLang: String): VocabularyPack {
        val sourcePack = LANGUAGE_TO_PACK[sourceLang] ?: "latin"
        val targetPack = LANGUAGE_TO_PACK[targetLang] ?: "latin"
        
        val packName = if (targetPack != "latin") targetPack else sourcePack
        
        return VocabularyPack(
            name = packName,
            languages = getLanguagesForPack(packName),
            downloadUrl = getDownloadUrl(packName),
            localPath = File(vocabDir, "$packName.msgpack").absolutePath,
            sizeMb = getPackSize(packName),
            version = "1.0"
        )
    }
    
    private fun getLanguagesForPack(packName: String): List<String> {
        return when (packName) {
            "latin" -> listOf("en", "es", "fr", "de", "it", "pt")
            "cjk" -> listOf("zh", "ja", "ko")
            "arabic" -> listOf("ar")
            "devanagari" -> listOf("hi")
            "cyrillic" -> listOf("ru", "uk")
            else -> listOf()
        }
    }
    
    private fun getDownloadUrl(packName: String): String {
        return "https://cdn.yourdomain.com/vocabs/${packName}_v1.0.msgpack"
    }
    
    private fun getPackSize(packName: String): Float {
        return when (packName) {
            "latin" -> 5.0f
            "cjk" -> 8.0f
            else -> 3.0f
        }
    }
}

// Main Translation Encoder
class TranslationEncoder(private val context: Context) {
    
    private var nativeHandle: Long = 0
    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .addInterceptor { chain ->
            val request = chain.request()
                .newBuilder()
                .addHeader("X-SDK-Version", "Android/1.0.0")
                .addHeader("X-SDK-Platform", "Android/${Build.VERSION.SDK_INT}")
                .addHeader("X-Device-Model", Build.MODEL)
                .build()
            Log.d(TAG, "Request: ${request.url}")
            chain.proceed(request)
        }
        .build()
    
    private val vocabularyManager = VocabularyManager(context)
    private val downloadScope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private val warmupScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    
    companion object {
        private const val TAG = "TranslationEncoder"
        
        init {
            try {
                System.loadLibrary("universal_encoder")
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Failed to load native library", e)
            }
        }
        
        // Native methods
        @JvmStatic
        private external fun nativeInit(modelPath: String): Long
        
        @JvmStatic
        private external fun nativeLoadVocabulary(handle: Long, vocabPath: String): Boolean
        
        @JvmStatic
        private external fun nativeEncode(
            handle: Long,
            text: String,
            sourceLang: String,
            targetLang: String
        ): ByteArray
        
        @JvmStatic
        private external fun nativeDestroy(handle: Long)
        
        @JvmStatic
        private external fun nativeGetMemoryUsage(handle: Long): Long
    }
    
    init {
        downloadScope.launch {
            initializeEncoder()
        }
        
        warmupScope.launch {
            delay(1000)
            warmupModel()
        }
    }
    
    private suspend fun warmupModel() = withContext(Dispatchers.Default) {
        try {
            if (nativeHandle != 0L) {
                Log.d(TAG, "Starting model warmup...")
                val startTime = System.currentTimeMillis()
                
                val dummyText = "test"
                nativeEncode(nativeHandle, dummyText, "en", "es")
                
                val warmupTime = System.currentTimeMillis() - startTime
                Log.d(TAG, "Model warmup completed in ${warmupTime}ms")
            }
        } catch (e: Exception) {
            Log.w(TAG, "Model warmup failed: ${e.message}")
        }
    }
    
    private suspend fun initializeEncoder() = withContext(Dispatchers.IO) {
        try {
            val modelFile = File(context.filesDir, "universal_encoder.onnx")
            if (!modelFile.exists()) {
                context.assets.open("models/universal_encoder_int8.onnx").use { input ->
                    modelFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
                Log.d(TAG, "Model extracted to: ${modelFile.absolutePath}")
            }
            
            nativeHandle = nativeInit(modelFile.absolutePath)
            if (nativeHandle == 0L) {
                Log.e(TAG, "Failed to initialize native encoder")
            } else {
                Log.d(TAG, "Native encoder initialized successfully")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize encoder", e)
        }
    }
    
    suspend fun prepareTranslation(sourceLang: String, targetLang: String): Boolean = 
        withContext(Dispatchers.IO) {
            try {
                val vocabPack = vocabularyManager.getVocabularyForPair(sourceLang, targetLang)
                
                if (vocabPack.needsDownload()) {
                    Log.d(TAG, "Downloading vocabulary pack: ${vocabPack.name}")
                    downloadVocabulary(vocabPack)
                }
                
                val loaded = nativeLoadVocabulary(nativeHandle, vocabPack.localPath)
                if (loaded) {
                    Log.d(TAG, "Vocabulary loaded successfully: ${vocabPack.name}")
                } else {
                    Log.e(TAG, "Failed to load vocabulary: ${vocabPack.name}")
                }
                
                loaded
            } catch (e: Exception) {
                Log.e(TAG, "Failed to prepare translation", e)
                false
            }
        }
    
    suspend fun encode(
        text: String,
        sourceLang: String,
        targetLang: String
    ): ByteArray = withContext(Dispatchers.Default) {
        PerformanceMonitor.measureTime("encode") {
            if (!prepareTranslation(sourceLang, targetLang)) {
                throw IOException("Failed to prepare translation")
            }
            
            val encoded = nativeEncode(nativeHandle, text, sourceLang, targetLang)
            Log.d(TAG, "Encoded ${text.length} chars to ${encoded.size} bytes")
            
            encoded
        }
    }
    
    private suspend fun downloadVocabulary(vocabPack: VocabularyPack) = withContext(Dispatchers.IO) {
        val request = Request.Builder()
            .url(vocabPack.downloadUrl)
            .build()
        
        httpClient.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw IOException("Failed to download vocabulary: ${response.code}")
            }
            
            response.body?.let { body ->
                val tempFile = File(vocabPack.localPath + ".tmp")
                tempFile.outputStream().use { output ->
                    body.byteStream().copyTo(output)
                }
                
                if (!tempFile.renameTo(File(vocabPack.localPath))) {
                    tempFile.delete()
                    throw IOException("Failed to save vocabulary pack")
                }
                
                Log.d(TAG, "Downloaded vocabulary pack: ${vocabPack.name} (${vocabPack.sizeMb}MB)")
            } ?: throw IOException("Empty response body")
        }
    }
    
    fun getMemoryUsage(): Long {
        return if (nativeHandle != 0L) {
            nativeGetMemoryUsage(nativeHandle)
        } else {
            0L
        }
    }
    
    fun getPerformanceMetrics(): List<PerformanceMetrics> {
        return PerformanceMonitor.getMetrics()
    }
    
    fun destroy() {
        downloadScope.cancel()
        warmupScope.cancel()
        if (nativeHandle != 0L) {
            nativeDestroy(nativeHandle)
            nativeHandle = 0
            Log.d(TAG, "Native encoder destroyed")
        }
    }
}