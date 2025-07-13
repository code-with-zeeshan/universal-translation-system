# data/practical_data_pipeline.py
class PracticalDataPipeline:
    """
    Realistic data pipeline for 20 languages
    Total data size: ~5-10GB (not 100s of GB!)
    """
    
    def prepare_all_data(self):
        # 1. Download high-quality evaluation sets (100MB total)
        self.download_eval_sets()
        
        # 2. Download English-centric pairs (2-3GB)
        self.download_english_pairs()
        
        # 3. Download key direct pairs (1-2GB)
        self.download_direct_pairs()
        
        # 4. Sample and filter large corpora (2-3GB)
        self.sample_large_corpora()
        
        # 5. Augment with synthetic data (1-2GB)
        self.augment_synthetic()
        
        print("✅ Total data: ~8GB of high-quality parallel text")
    
    def get_training_distribution(self):
        """Smart data distribution for training"""
        return {
            # High-resource pairs (more data)
            'en-es': 2000000,  # 2M sentences
            'en-fr': 2000000,
            'en-de': 2000000,
            'en-zh': 1500000,
            'en-ru': 1500000,
            
            # Medium-resource pairs
            'en-ja': 1000000,
            'en-ar': 1000000,
            'en-pt': 1000000,
            'en-it': 1000000,
            
            # Low-resource pairs (less data)
            'en-hi': 500000,
            'en-ko': 500000,
            'en-tr': 500000,
            'en-th': 300000,
            'en-vi': 300000,
            'en-pl': 300000,
            'en-uk': 300000,
            'en-nl': 300000,
            'en-id': 300000,
            'en-sv': 300000,
            
            # Direct pairs (supplementary)
            'es-pt': 200000,
            'zh-ja': 200000,
            'fr-es': 200000,
            'de-fr': 200000,
            'ru-uk': 200000,
        }