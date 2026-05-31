// android/UniversalTranslationSDK/src/main/java/com/universaltranslation/encoder/MainActivity.kt
package com.universaltranslation.encoder

import android.os.Bundle
import android.util.Log
import android.widget.TextView
import android.widget.Toast
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.work.WorkInfo
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import com.google.android.material.textfield.TextInputEditText
import com.universaltranslation.sdk.R
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {
    companion object {
        private const val TAG = "MainActivity"
    }
    
    private lateinit var translationClient: TranslationClient
    private lateinit var vocabManager: EnhancedVocabularyPackManager
    
    // UI components
    private lateinit var inputText: TextInputEditText
    private lateinit var outputText: TextView
    private lateinit var translateButton: MaterialButton
    private lateinit var statusText: TextView
    private lateinit var sourceLanguageSpinner: Spinner
    private lateinit var targetLanguageSpinner: Spinner
    private lateinit var swapLanguagesButton: ImageButton
    private lateinit var outputCard: MaterialCardView
    
    // Language arrays
    private lateinit var languageNames: Array<String>
    private lateinit var languageCodes: Array<String>
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        // Initialize UI components
        initializeUI()
        
        // Initialize translation components
        translationClient = TranslationClient(this)
        vocabManager = EnhancedVocabularyPackManager(this)
        
        // Setup spinners
        setupLanguageSpinners()

        // Preload user's preferred languages
        preloadLanguages()
        
        // Setup button click
        translateButton.setOnClickListener {
            translateText()
        }

        swapLanguagesButton.setOnClickListener {
            swapLanguages()
        }
    }
    
    private fun initializeUI() {
        inputText = findViewById(R.id.inputText)
        outputText = findViewById(R.id.outputText)
        translateButton = findViewById(R.id.translateButton)
        statusText = findViewById(R.id.statusText)
        sourceLanguageSpinner = findViewById(R.id.sourceLanguageSpinner)
        targetLanguageSpinner = findViewById(R.id.targetLanguageSpinner)
        swapLanguagesButton = findViewById(R.id.swapLanguagesButton)
        outputCard = findViewById(R.id.outputCard)
    }

    private fun setupLanguageSpinners() {
        // Set default selections
        sourceLanguageSpinner.setSelection(0) // English
        targetLanguageSpinner.setSelection(1) // Spanish
        
        // Add listeners
        sourceLanguageSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                // Ensure source and target are different
                if (position == targetLanguageSpinner.selectedItemPosition) {
                    targetLanguageSpinner.setSelection(if (position == 0) 1 else 0)
                }
            }
            
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
        
        targetLanguageSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                // Ensure source and target are different
                if (position == sourceLanguageSpinner.selectedItemPosition) {
                    sourceLanguageSpinner.setSelection(if (position == 0) 1 else 0)
                }
            }
            
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
    }
    
    private fun swapLanguages() {
        val sourcePos = sourceLanguageSpinner.selectedItemPosition
        val targetPos = targetLanguageSpinner.selectedItemPosition
        
        sourceLanguageSpinner.setSelection(targetPos)
        targetLanguageSpinner.setSelection(sourcePos)
        
        // If there's a translation, swap the texts
        if (outputText.text.isNotEmpty() && outputText.text != getString(R.string.translation_will_appear_here)) {
            inputText.setText(outputText.text)
            outputText.text = getString(R.string.translation_will_appear_here)
        }
    }
    
    private fun preloadLanguages() {
        val userLanguages = setOf("en", "es", "fr")
        vocabManager.downloadPacksForLanguages(userLanguages)
        
        // Observe download progress
        vocabManager.getDownloadProgress().observe(this) { workInfos ->
            workInfos.forEach { workInfo ->
                when (workInfo.state) {
                    WorkInfo.State.RUNNING -> {
                        val progress = workInfo.progress.getInt("progress", 0)
                        statusText.text = "Downloading vocabularies: $progress%"
                        Log.d(TAG, "Download progress: $progress%")
                    }
                    WorkInfo.State.SUCCEEDED -> {
                        statusText.text = "Vocabularies ready"
                        Log.d(TAG, "Download completed")
                    }
                    WorkInfo.State.FAILED -> {
                        statusText.text = "Download failed"
                        Log.e(TAG, "Download failed")
                    }
                    else -> {
                        // Handle other states if needed
                    }
                }
            }
        }
    }
    
    private fun translateText() {
        val text = inputText.text.toString()
        if (text.isBlank()) {
            Toast.makeText(this, "Please enter text to translate", Toast.LENGTH_SHORT).show()
            return
        }

        // Get selected languages
        val sourceLang = languageCodes[sourceLanguageSpinner.selectedItemPosition]
        val targetLang = languageCodes[targetLanguageSpinner.selectedItemPosition]
        
        // Disable button during translation
        translateButton.isEnabled = false
        statusText.text = getString(R.string.translating)
        
        lifecycleScope.launch {
            when (val result = translationClient.translate(text, "en", "es")) {
                is TranslationResult.Success -> {
                    outputText.text = result.translation
                    statusText.text = "Translation complete"
                    
                    // Show performance metrics
                    val metrics = translationClient.getPerformanceMetrics()
                    metrics.lastOrNull()?.let { metric ->
                        Log.d(TAG, "Translation took ${metric.duration}ms")
                    }
                }
                is TranslationResult.Error -> {
                    outputText.text = ""
                    statusText.text = "Error: ${result.message}"
                    Toast.makeText(this@MainActivity, result.message, Toast.LENGTH_LONG).show()
                }
            }
            
            // Re-enable button
            translateButton.isEnabled = true
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        translationClient.destroy()
        vocabManager.cancelAllDownloads()
    }
}