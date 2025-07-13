// android/UniversalTranslationSDK/src/main/java/com/universaltranslation/encoder/TranslationEncoder.kt
package com.universaltranslation.encoder

import android.content.Context
import kotlinx.coroutines.*
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.util.concurrent.TimeUnit

class TranslationEncoder(private val context: Context) {
    
    private var nativeHandle: Long = 0
    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .build()
    
    private val vocabularyManager = VocabularyManager(context)
    
    companion object {
        init {
            System.loadLibrary("universal_encoder")
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
    }
    
    init {
        initializeEncoder()
    }
    
    private fun initializeEncoder() {
        // Extract model from assets if needed
        val modelFile = File(context.filesDir, "universal_encoder.onnx")
        if (!modelFile.exists()) {
            context.assets.open("models/universal_encoder.onnx").use { input ->
                modelFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }
        
        nativeHandle = nativeInit(modelFile.absolutePath)
    }
    
    suspend fun prepareTranslation(sourceLang: String, targetLang: String): Boolean = withContext(Dispatchers.IO) {
        // Download vocabulary if needed
        val vocabPack = vocabularyManager.getVocabularyForPair(sourceLang, targetLang)
        
        if (vocabPack.needsDownload()) {
            downloadVocabulary(vocabPack)
        }
        
        // Load into native encoder
        nativeLoadVocabulary(nativeHandle, vocabPack.localPath)
    }
    
    suspend fun encode(
        text: String,
        sourceLang: String,
        targetLang: String
    ): ByteArray = withContext(Dispatchers.Default) {
        
        // Ensure vocabulary is loaded
        prepareTranslation(sourceLang, targetLang)
        
        // Encode using native method
        nativeEncode(nativeHandle, text, sourceLang, targetLang)
    }
    
    private suspend fun downloadVocabulary(vocabPack: VocabularyPack) = withContext(Dispatchers.IO) {
        val request = Request.Builder()
            .url(vocabPack.downloadUrl)
            .build()
        
        httpClient.newCall(request).execute().use { response ->
            if (response.isSuccessful) {
                response.body?.let { body ->
                    File(vocabPack.localPath).outputStream().use { output ->
                        body.byteStream().copyTo(output)
                    }
                }
            }
        }
    }
    
    fun destroy() {
        if (nativeHandle != 0L) {
            nativeDestroy(nativeHandle)
            nativeHandle = 0
        }
    }
}

// Complete Translation Client
class TranslationClient(
    private val context: Context,
    private val decoderUrl: String = "https://api.yourdomain.com/decode"
) {
    
    private val encoder = TranslationEncoder(context)
    private val httpClient = OkHttpClient()
    private val gson = Gson()
    
    suspend fun translate(
        text: String,
        sourceLang: String,
        targetLang: String
    ): TranslationResult = withContext(Dispatchers.IO) {
        
        try {
            // Encode locally
            val encoded = encoder.encode(text, sourceLang, targetLang)
            
            // Send to decoder
            val request = Request.Builder()
                .url(decoderUrl)
                .post(
                    RequestBody.create(
                        MediaType.parse("application/octet-stream"),
                        encoded
                    )
                )
                .header("X-Target-Language", targetLang)
                .build()
            
            httpClient.newCall(request).execute().use { response ->
                if (response.isSuccessful) {
                    val result = gson.fromJson(
                        response.body?.string(),
                        TranslationResponse::class.java
                    )
                    TranslationResult.Success(result.translation)
                } else {
                    TranslationResult.Error("Translation failed: ${response.code}")
                }
            }
        } catch (e: Exception) {
            TranslationResult.Error(e.message ?: "Unknown error")
        }
    }
}

// Usage
class MainActivity : AppCompatActivity() {
    private lateinit var translationClient: TranslationClient
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        translationClient = TranslationClient(this)
        
        // Translate text
        lifecycleScope.launch {
            when (val result = translationClient.translate("Hello world", "en", "es")) {
                is TranslationResult.Success -> {
                    showTranslation(result.translation)
                }
                is TranslationResult.Error -> {
                    showError(result.message)
                }
            }
        }
    }
}