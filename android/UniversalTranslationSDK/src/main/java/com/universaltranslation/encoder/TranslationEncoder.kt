// android/UniversalTranslationSDK/src/main/java/com/universaltranslation/encoder/TranslationEncoder.kt
package com.universaltranslation.encoder

import android.content.Context
import androidx.lifecycle.lifecycleScope
import androidx.appcompat.app.AppCompatActivity
import com.google.gson.Gson
import kotlinx.coroutines.*
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.File
import java.util.concurrent.TimeUnit
import android.util.Log
import androidx.annotation.Keep
import java.io.IOException

// Data classes
@Keep
data class VocabularyPack(
    val name: String,
    val languages: List<String>,
    val downloadUrl: String,
    val localPath: String,
    val sizeMb: Float,
    val version: String
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

// Vocabulary Manager
class VocabularyManager(private val context: Context) {
    
    companion object {
        private const val TAG = "VocabularyManager"
        private const val VOCAB_DIR = "vocabularies"
        private val LANGUAGE_TO_PACK = mapOf(
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
        
        // Use target language pack as priority
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
        // Replace with your actual CDN URL
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
            Log.d(TAG, "Request: ${request.url}")
            chain.proceed(request)
        }
        .build()
    
    private val vocabularyManager = VocabularyManager(context)
    private val downloadScope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    
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
    }
    
    private suspend fun initializeEncoder() = withContext(Dispatchers.IO) {
        try {
            // Extract model from assets if needed
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
    
    suspend fun prepareTranslation(sourceLang: String, targetLang: String): Boolean = withContext(Dispatchers.IO) {
        try {
            // Get vocabulary pack
            val vocabPack = vocabularyManager.getVocabularyForPair(sourceLang, targetLang)
            
            // Download if needed
            if (vocabPack.needsDownload()) {
                Log.d(TAG, "Downloading vocabulary pack: ${vocabPack.name}")
                downloadVocabulary(vocabPack)
            }
            
            // Load into native encoder
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
        // Ensure vocabulary is loaded
        if (!prepareTranslation(sourceLang, targetLang)) {
            throw IOException("Failed to prepare translation")
        }
        
        // Encode using native method
        val encoded = nativeEncode(nativeHandle, text, sourceLang, targetLang)
        Log.d(TAG, "Encoded ${text.length} chars to ${encoded.size} bytes")
        
        encoded
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
                
                // Atomic rename
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
    
    fun destroy() {
        downloadScope.cancel()
        if (nativeHandle != 0L) {
            nativeDestroy(nativeHandle)
            nativeHandle = 0
            Log.d(TAG, "Native encoder destroyed")
        }
    }
}

// Translation Client
class TranslationClient(
    private val context: Context,
    private val decoderUrl: String = "https://api.yourdomain.com/decode"
) {
    companion object {
        private const val TAG = "TranslationClient"
    }
    
    private val encoder = TranslationEncoder(context)
    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .build()
    private val gson = Gson()
    
    suspend fun translate(
        text: String,
        sourceLang: String,
        targetLang: String
    ): TranslationResult = withContext(Dispatchers.IO) {
        try {
            // Validate input
            if (text.isBlank()) {
                return@withContext TranslationResult.Error("Text cannot be empty")
            }
            
            // Encode locally
            val encoded = encoder.encode(text, sourceLang, targetLang)
            
            // Send to decoder
            val requestBody = encoded.toRequestBody("application/octet-stream".toMediaType())
            
            val request = Request.Builder()
                .url(decoderUrl)
                .post(requestBody)
                .header("X-Target-Language", targetLang)
                .header("X-Source-Language", sourceLang)
                .header("Content-Type", "application/octet-stream")
                .build()
            
            httpClient.newCall(request).execute().use { response ->
                when {
                    response.isSuccessful -> {
                        val responseBody = response.body?.string() ?: throw IOException("Empty response")
                        val result = gson.fromJson(responseBody, TranslationResponse::class.java)
                        TranslationResult.Success(result.translation)
                    }
                    response.code == 429 -> {
                        TranslationResult.Error("Rate limit exceeded. Please try again later.")
                    }
                    response.code in 500..599 -> {
                        TranslationResult.Error("Server error. Please try again later.")
                    }
                    else -> {
                        TranslationResult.Error("Translation failed: ${response.code}")
                    }
                }
            }
        } catch (e: IOException) {
            Log.e(TAG, "Network error", e)
            TranslationResult.Error("Network error: ${e.message}")
        } catch (e: Exception) {
            Log.e(TAG, "Translation error", e)
            TranslationResult.Error("Translation error: ${e.message}")
        }
    }
    
    fun destroy() {
        encoder.destroy()
    }
}

// Usage Example Activity
class MainActivity : AppCompatActivity() {
    companion object {
        private const val TAG = "MainActivity"
    }
    
    private lateinit var translationClient: TranslationClient
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        translationClient = TranslationClient(this)
        
        // Example translation
        translateText("Hello world", "en", "es")
    }
    
    private fun translateText(text: String, sourceLang: String, targetLang: String) {
        lifecycleScope.launch {
            when (val result = translationClient.translate(text, sourceLang, targetLang)) {
                is TranslationResult.Success -> {
                    showTranslation(result.translation)
                }
                is TranslationResult.Error -> {
                    showError(result.message)
                }
            }
        }
    }
    
    private fun showTranslation(translation: String) {
        Log.d(TAG, "Translation: $translation")
        // Update UI
    }
    
    private fun showError(message: String) {
        Log.e(TAG, "Error: $message")
        // Show error to user
    }
    
    override fun onDestroy() {
        super.onDestroy()
        translationClient.destroy()
    }
}

// Vocabulary Pack Manager for bulk operations
class VocabularyPackManager(private val context: Context) {
    private val vocabularyManager = VocabularyManager(context)
    private val workManager = androidx.work.WorkManager.getInstance(context)
    
    fun downloadPacksForLanguages(languages: Set<String>) {
        val requiredPacks = languages.mapNotNull { lang ->
            LANGUAGE_TO_PACK[lang]
        }.toSet()
        
        requiredPacks.forEach { packName ->
            schedulePackDownload(packName)
        }
    }
    
    private fun schedulePackDownload(packName: String) {
        val downloadRequest = androidx.work.OneTimeWorkRequestBuilder<VocabularyDownloadWorker>()
            .setInputData(
                androidx.work.Data.Builder()
                    .putString("pack_name", packName)
                    .build()
            )
            .setConstraints(
                androidx.work.Constraints.Builder()
                    .setRequiredNetworkType(androidx.work.NetworkType.CONNECTED)
                    .build()
            )
            .build()
        
        workManager.enqueue(downloadRequest)
    }
    
    companion object {
        private val LANGUAGE_TO_PACK = mapOf(
            "en" to "latin", "es" to "latin", "fr" to "latin",
            "de" to "latin", "it" to "latin", "pt" to "latin",
            "zh" to "cjk", "ja" to "cjk", "ko" to "cjk",
            "ar" to "arabic", "hi" to "devanagari",
            "ru" to "cyrillic", "uk" to "cyrillic"
        )
    }
}

// Background download worker
class VocabularyDownloadWorker(
    context: Context,
    params: androidx.work.WorkerParameters
) : androidx.work.CoroutineWorker(context, params) {
    
    override suspend fun doWork(): Result {
        val packName = inputData.getString("pack_name") ?: return Result.failure()
        
        return try {
            // Download vocabulary pack
            // Implementation depends on your download mechanism
            Result.success()
        } catch (e: Exception) {
            Result.retry()
        }
    }
}