## 📋 **Adding New Languages - Complete Process**

### **Step-by-Step Process for Adding Languages (e.g., Adding Portuguese-Brazil 'pt-BR')**

| Step | Component | Action Required | Code/Command | Time |
|------|-----------|----------------|--------------|------|
| **1** | **Config** | Add to language list | ```yaml<br># data/config.yaml<br>languages:<br>  - pt-BR  # Add here<br>``` | 5 min |
| **2** | **Data Collection** | Gather corpus data | ```bash<br># Add to corpus_paths<br>'pt-BR': 'data/pt-BR_corpus.txt'<br>``` | 1-2 days |
| **3** | **Data Pipeline** | Process new data | ```python<br># Add to training_distribution<br>training_distribution:<br>  en-pt-BR: 500000<br>  pt-pt-BR: 200000<br>``` | 2-4 hours |
| **4** | **Vocabulary Decision** | Choose vocabulary strategy | **Option A**: Add to existing Latin pack<br>**Option B**: Create new pack | 30 min |
| **5** | **Vocabulary Creation** | Generate/Update pack | ```bash<br># If adding to Latin pack<br>python vocabulary/create_vocabulary_packs_from_data.py \<br>  --update-pack latin \<br>  --add-language pt-BR<br>``` | 2-3 hours |
| **6** | **Model Embeddings** | Initialize new embeddings | ```python<br># Auto-handled if using dynamic loading<br># Otherwise: initialize embeddings for new tokens<br>``` | 1 hour |
| **7** | **Testing** | Validate integration | ```python<br># Test new language pair<br>test_translation('en', 'pt-BR', 'Hello')<br>``` | 1 hour |
| **8** | **Deployment** | Update production | Deploy new vocabulary pack only | 30 min |

### **Language Addition Scenarios**

| Scenario | Languages | Vocabulary Strategy | Complexity | Time |
|----------|-----------|-------------------|------------|------|
| **Similar Script** | Adding Italian to Latin group | Update existing Latin pack | Low | 1 day |
| **New Script** | Adding Tamil (new script) | Create new Dravidian pack | High | 3-5 days |
| **Rare Language** | Adding Quechua | Create specialized pack or add to Latin | Medium | 2-3 days |
| **Dialect** | Adding Swiss German | Extend German vocabulary | Low | 1 day |
| **Code-Mixed** | Adding Hinglish (Hindi+English) | Create hybrid pack | High | 4-5 days |