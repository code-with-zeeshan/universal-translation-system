// android/UniversalTranslationSDK/src/main/java/com/universaltranslation/encoder/TranslationClient.kt
package com.universaltranslation.encoder

import android.content.Context
import android.util.Log
import com.google.gson.Gson
import kotlinx.coroutines.*
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException
import java.util.concurrent.TimeUnit

// Error codes enum
enum class TranslationErrorCode(val code: String) {
    NETWORK_ERROR("NETWORK_ERROR"),
    MODEL_NOT_FOUND("MODEL_NOT_FOUND"),  
    VOCABULARY_NOT_LOADED("VOCABULARY_NOT_LOADED"),
    ENCODING_FAILED("ENCODING_FAILED"),
    DECODING_FAILED("DECODING_FAILED"),
    INVALID_LANGUAGE("INVALID_LANGUAGE"),
    RATE_LIMITED("RATE_LIMITED")
}

// Analytics class
class TranslationAnalytics {
    fun trackTranslation(sourceLang: String, targetLang: String, textLength: Int, duration: Long) {
        // Implement your analytics tracking
        // Example: Firebase Analytics, Amplitude, etc.
        Log.d("Analytics", "Translation completed: $sourceLang->$targetLang, ${textLength} chars, ${duration}ms")
    }
    
    fun trackError(error: TranslationErrorCode, context: Map<String, Any>) {
        Log.e("Analytics", "Translation error: ${error.code}, context: $context")
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
    private val analytics = TranslationAnalytics()
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
        val startTime = System.currentTimeMillis()
        
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
                val result = when {
                    response.isSuccessful -> {
                        val responseBody = response.body?.string() ?: throw IOException("Empty response")
                        val translationResponse = gson.fromJson(responseBody, TranslationResponse::class.java)
                        TranslationResult.Success(translationResponse.translation)
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
                
                // Track analytics
                val duration = System.currentTimeMillis() - startTime
                when (result) {
                    is TranslationResult.Success -> {
                        analytics.trackTranslation(sourceLang, targetLang, text.length, duration)
                    }
                    is TranslationResult.Error -> {
                        analytics.trackError(
                            getErrorCode(response.code),
                            mapOf(
                                "source_lang" to sourceLang,
                                "target_lang" to targetLang,
                                "error_message" to result.message
                            )
                        )
                    }
                }
                
                result
            }
        } catch (e: IOException) {
            Log.e(TAG, "Network error", e)
            val duration = System.currentTimeMillis() - startTime
            analytics.trackError(
                TranslationErrorCode.NETWORK_ERROR,
                mapOf(
                    "source_lang" to sourceLang,
                    "target_lang" to targetLang,
                    "error_message" to (e.message ?: "Unknown error"),
                    "duration" to duration
                )
            )
            TranslationResult.Error("Network error: ${e.message}")
        } catch (e: Exception) {
            Log.e(TAG, "Translation error", e)
            val duration = System.currentTimeMillis() - startTime
            analytics.trackError(
                TranslationErrorCode.ENCODING_FAILED,
                mapOf(
                    "source_lang" to sourceLang,
                    "target_lang" to targetLang,
                    "error_message" to (e.message ?: "Unknown error"),
                    "duration" to duration
                )
            )
            TranslationResult.Error("Translation error: ${e.message}")
        }
    }
    
    private fun getErrorCode(httpCode: Int): TranslationErrorCode {
        return when (httpCode) {
            429 -> TranslationErrorCode.RATE_LIMITED
            in 500..599 -> TranslationErrorCode.NETWORK_ERROR
            else -> TranslationErrorCode.DECODING_FAILED
        }
    }
    
    fun getPerformanceMetrics(): List<PerformanceMetrics> {
        return encoder.getPerformanceMetrics()
    }
    
    fun destroy() {
        encoder.destroy()
    }
}