// android/UniversalTranslationSDK/src/main/java/com/universaltranslation/encoder/VocabularyPackManager.kt
package com.universaltranslation.encoder

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.Build
import android.util.Log
import android.util.LruCache
import androidx.lifecycle.LiveData
import androidx.work.*
import java.io.File
import java.io.RandomAccessFile
import java.nio.channels.FileChannel
import java.util.concurrent.TimeUnit
import java.io.IOException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import androidx.work.Worker
import androidx.work.WorkerParameters
import org.msgpack.core.MessagePack

class EnhancedVocabularyPackManager(private val context: Context) {
    
    companion object {
        private const val TAG = "VocabularyPackManager"
        private const val WORK_TAG = "vocabulary_download"
        private const val CACHE_SIZE = 3
        private const val LARGE_FILE_THRESHOLD = 5 * 1024 * 1024 // 5MB
    }
    
    private val vocabularyManager = VocabularyManager(context)
    private val workManager = WorkManager.getInstance(context)
    private val vocabCache = LruCache<String, VocabularyPack>(CACHE_SIZE)
    
    fun downloadPacksForLanguages(languages: Set<String>) {
        val connectivityManager = context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        
        // Calculate total download size
        val totalSize = calculateTotalSize(languages)
        Log.d(TAG, "Total download size: ${totalSize / 1024 / 1024}MB for ${languages.size} languages")
        
        // Build constraints based on network and size
        val constraints = Constraints.Builder().apply {
            // Use WiFi for large downloads
            if (totalSize > 10 * 1024 * 1024) { // 10MB
                setRequiredNetworkType(NetworkType.UNMETERED)
            } else {
                setRequiredNetworkType(NetworkType.CONNECTED)
            }
            
            // Don't download when battery is low
            setRequiresBatteryNotLow(true)
            
            // Require storage not low
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                setRequiresStorageNotLow(true)
            }
        }.build()
        
        // Schedule downloads
        languages.forEach { language ->
            schedulePackDownload(language, constraints)
        }
    }
    
    fun loadVocabularyEfficient(packName: String): VocabularyPack? {
        // Check memory cache first
        vocabCache.get(packName)?.let {
            Log.d(TAG, "Vocabulary loaded from cache: $packName")
            return it
        }
        
        val vocabDir = File(context.filesDir, VocabularyManager.VOCAB_DIR)
        val file = File(vocabDir, "$packName.msgpack")
        
        if (!file.exists()) {
            Log.w(TAG, "Vocabulary file not found: $packName")
            return null
        }
        
        return try {
            val pack = if (file.length() > LARGE_FILE_THRESHOLD) {
                Log.d(TAG, "Using memory-mapped loading for large file: $packName (${file.length() / 1024 / 1024}MB)")
                loadVocabularyMemoryMapped(file)
            } else {
                Log.d(TAG, "Loading vocabulary normally: $packName")
                parseVocabularyFromFile(file)
            }
            
            // Cache the loaded pack
            vocabCache.put(packName, pack)
            pack
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load vocabulary: $packName", e)
            null
        }
    }
    
    private fun loadVocabularyMemoryMapped(file: File): VocabularyPack {
        RandomAccessFile(file, "r").use { raf ->
            val channel = raf.channel
            val buffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
            
            // Parse MessagePack data from memory-mapped buffer
            val data = ByteArray(buffer.remaining())
            buffer.get(data)
            
            // Use your existing parsing logic
            return parseMessagePackData(data, file)
        }
    }
    
    private fun parseVocabularyFromFile(file: File): VocabularyPack {
        // Your existing file parsing logic
        return VocabularyPack(
            name = file.nameWithoutExtension,
            languages = getLanguagesForPack(file.nameWithoutExtension),
            downloadUrl = "",
            localPath = file.absolutePath,
            sizeMb = (file.length() / 1024f / 1024f),
            version = "1.0"
        )
    }
    
    private fun parseMessagePackData(data: ByteArray, file: File): VocabularyPack {
        // Production: Parse MessagePack data using a library
        // Example using msgpack-java (add to build.gradle):
        // implementation 'org.msgpack:msgpack-core:0.8.22'
        val msgpack = org.msgpack.core.MessagePack.newDefaultUnpacker(data)
        val tokens = mutableMapOf<String, Int>()
        // Parse tokens map (simplified, adjust for your schema)
        msgpack.unpackMapHeader()
        while (msgpack.hasNext()) {
            val key = msgpack.unpackString()
            val value = msgpack.unpackInt()
            tokens[key] = value
        }
        return VocabularyPack(
            name = file.nameWithoutExtension,
            languages = getLanguagesForPack(file.nameWithoutExtension),
            downloadUrl = "",
            localPath = file.absolutePath,
            sizeMb = (file.length() / 1024f / 1024f),
            version = "1.0",
            tokens = tokens
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
    
    private fun calculateTotalSize(languages: Set<String>): Long {
        var totalSize = 0L
        val requiredPacks = languages.map { lang ->
            VocabularyManager.LANGUAGE_TO_PACK[lang] ?: "latin"
        }.toSet()
        
        requiredPacks.forEach { packName ->
            totalSize += when (packName) {
                "latin" -> 5 * 1024 * 1024
                "cjk" -> 8 * 1024 * 1024
                "arabic" -> 3 * 1024 * 1024
                "cyrillic" -> 4 * 1024 * 1024
                else -> 3 * 1024 * 1024
            }
        }
        
        return totalSize
    }
    
    private fun schedulePackDownload(language: String, constraints: Constraints) {
        val packName = VocabularyManager.LANGUAGE_TO_PACK[language] ?: "latin"
        val inputData = workDataOf(
            "language" to language,
            "pack_name" to packName
        )
        
        val downloadRequest = OneTimeWorkRequestBuilder<VocabularyDownloadWorker>()
            .setInputData(inputData)
            .setConstraints(constraints)
            .addTag(WORK_TAG)
            .setBackoffCriteria(
                BackoffPolicy.EXPONENTIAL,
                WorkRequest.MIN_BACKOFF_MILLIS,
                TimeUnit.MILLISECONDS
            )
            .build()
        
        workManager.enqueueUniqueWork(
            "download_vocab_$language",
            ExistingWorkPolicy.KEEP,
            downloadRequest
        )
    }
    
    fun cancelAllDownloads() {
        workManager.cancelAllWorkByTag(WORK_TAG)
    }
    
    fun getDownloadProgress(): LiveData<List<WorkInfo>> {
        return workManager.getWorkInfosByTagLiveData(WORK_TAG)
    }
}

// Background download worker implementation
class VocabularyDownloadWorker(
    context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {
    
    companion object {
        private const val TAG = "VocabularyDownloadWorker"
    }
    
    override suspend fun doWork(): Result {
        val packName = inputData.getString("pack_name") ?: return Result.failure()
        val language = inputData.getString("language") ?: return Result.failure()
        
        return try {
            Log.d(TAG, "Starting download for $packName (language: $language)")
            
            // Get vocabulary manager
            val vocabManager = VocabularyManager(applicationContext)
            val vocabPack = vocabManager.getVocabularyForPair(language, language)
            
            // Download vocabulary pack
            downloadVocabularyPack(vocabPack)
            
            Log.d(TAG, "Successfully downloaded $packName")
            Result.success()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to download $packName", e)
            if (runAttemptCount < 3) {
                Result.retry()
            } else {
                Result.failure()
            }
        }
    }
    
    private suspend fun downloadVocabularyPack(vocabPack: VocabularyPack) = withContext(Dispatchers.IO) {
        val client = OkHttpClient.Builder()
            .connectTimeout(60, TimeUnit.SECONDS)
            .readTimeout(60, TimeUnit.SECONDS)
            .build()
        
        val request = Request.Builder()
            .url(vocabPack.downloadUrl)
            .build()
        
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw IOException("Failed to download: ${response.code}")
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
            } ?: throw IOException("Empty response body")
        }
    }
}