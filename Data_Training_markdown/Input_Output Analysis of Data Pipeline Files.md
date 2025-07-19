## Input/Output Analysis of Data Pipeline Files

### 📊 Core Module I/O Table

| File | Inputs | Outputs | Output Format |
|------|---------|----------|---------------|
| **config.yaml** | None | • Language list (20 languages)<br>• Training distribution<br>• Quality parameters<br>• Directory paths | YAML configuration |
| **smart_data_downloader.py** | • config.yaml<br>• None (standalone) | • List of `LanguagePair` objects<br>• Download schedule<br>• Size estimates<br>• Strategy JSON | • Python objects<br>• JSON file (`download_strategy.json`) |
| **download_curated_data.py** | • Language list<br>• Internet connection | • FLORES-200 dataset<br>• Tatoeba pairs<br>• OpenSubtitles samples<br>• MultiUN samples | • HF Dataset format<br>• Text files (.txt)<br>• Zipped files |
| **download_training_data.py** | • Language list<br>• Internet connection | • OPUS datasets<br>• WMT datasets<br>• NLLB seed data<br>• CCMatrix data | • HF Dataset format<br>• Batch files |
| **smart_sampler.py** | • Raw text file (tab-separated)<br>• Target sample size<br>• Quality thresholds | • Filtered sentence pairs<br>• Sampling statistics | • Tab-separated text file<br>• Statistics dict |
| **synthetic_augmentation.py** | • Monolingual text files<br>• Parallel text files<br>• Language codes | • Backtranslated pairs<br>• Pivot translations<br>• Augmentation statistics | • Tab-separated text files<br>• Statistics dict |
| **practical_data_pipeline.py** | • config.yaml<br>• All sub-modules | • Complete dataset<br>• Validation report | • Organized directory structure<br>• Log files |

### 📁 Data Flow Details

| Stage | Input Files/Data | Processing | Output Files | Size |
|-------|------------------|------------|--------------|------|
| **1. Configuration** | `config.yaml` | Parse YAML | Python dict | ~2KB |
| **2. Strategy** | Config data | Generate language pairs | `download_strategy.json` | ~10KB |
| **3. Essential Data** | Web APIs | Download + Save | `data/essential/`<br>• `flores200/`<br>• `tatoeba_en_*.txt`<br>• `OpenSubtitles_*.txt` | ~100MB |
| **4. Training Data** | Web APIs | Stream + Batch | `data/raw/`<br>• `opus/`<br>• `wmt/`<br>• `nllb_*.txt` | ~50GB |
| **5. Sampling** | `data/raw/*.txt` | Filter + Sample | `data/sampled/`<br>• `en-es_sampled.txt`<br>• `en-fr_sampled.txt`<br>• etc. | ~5GB |
| **6. Augmentation** | • `data/raw/mono_*.txt`<br>• `data/sampled/*.txt` | Backtranslate + Pivot | `data/final/`<br>• `augmented_*.txt`<br>• `pivot_pairs/*.txt` | ~3GB |
| **7. Final Output** | All processed data | Validate + Merge | `data/processed/`<br>• Final training corpus | ~8GB |

### 🔄 File Format Specifications

| File Type | Format | Example | Fields |
|-----------|---------|---------|---------|
| **Parallel Text** | Tab-separated | `Hello world\tHola mundo` | source_text \t target_text |
| **Monolingual Text** | One per line | `This is a sentence.` | text |
| **HF Dataset** | Arrow format | Binary files | Depends on dataset |
| **Config YAML** | YAML | `languages: [en, es, fr]` | Key-value pairs |
| **Strategy JSON** | JSON | `{"pairs": [...]}` | Structured data |
| **Log Files** | Text with timestamps | `2025-01-01 12:00:00 - INFO - Message` | timestamp - level - message |

### 📋 Utility Module I/O

| Module | Function | Input | Output |
|--------|----------|--------|---------|
| **data_utils.py** | `ConfigManager.load_config()` | YAML path | Config dict |
| | `DataProcessor.process_streaming_dataset()` | HF Dataset | Saved batches |
| | `estimate_sentence_count()` | Text file path | Integer count |
| | `merge_datasets()` | List of file paths | Single merged file |
| **common_utils.py** | `DirectoryManager.create_directory()` | Path string | Path object |
| | `StandardLogger.get_logger()` | Logger name | Logger instance |
| | `ImportCleaner.get_recommended_imports()` | Module type | Import statements |

### 🎯 Expected Data Volumes

| Language Pair | Input Sources | Expected Output | Sentences |
|---------------|---------------|-----------------|-----------|
| en-es | OPUS, NLLB, WMT | 2GB | 2,000,000 |
| en-fr | OPUS, NLLB, WMT | 2GB | 2,000,000 |
| en-de | OPUS, NLLB, WMT | 2GB | 2,000,000 |
| en-zh | OPUS, NLLB, CCMatrix | 1.5GB | 1,500,000 |
| en-ru | OPUS, NLLB, CCMatrix | 1.5GB | 1,500,000 |
| Other pairs | Various | 0.1-1GB each | 200K-1M |

This table structure clearly shows what each module takes as input and what it produces as output, making the data flow through the pipeline transparent and easy to understand.