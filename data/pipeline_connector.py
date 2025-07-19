# data/pipeline_connector.py
"""Connects data pipeline → vocabulary creation → training"""
from pathlib import Path
import logging
from typing import List, Dict
from tqdm import tqdm

from data_utils import ConfigManager, merge_datasets
from utils.common_utils import StandardLogger

class PipelineConnector:
    """Connects all pipeline stages"""
    
    def __init__(self):
        self.logger = StandardLogger.get_logger(__name__)
        self.config = ConfigManager.load_config()
        
    def create_monolingual_corpora(self):
        """Split parallel data into monolingual files for vocabulary creation"""
        sampled_dir = Path(self.config['output_dir']) / 'sampled'
        processed_dir = Path(self.config['output_dir']) / 'processed'
        processed_dir.mkdir(exist_ok=True)
        
        language_texts = {}
        
        # Read all sampled files
        for file_path in sampled_dir.glob('*_sampled.txt'):
            self.logger.info(f"Processing {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=f"Reading {file_path.name}"):
                    parts = line.strip().split('\t')
                    if len(parts) == 4:  # source, target, source_lang, target_lang
                        source_text, target_text, source_lang, target_lang = parts
                        
                        # Collect texts by language
                        if source_lang not in language_texts:
                            language_texts[source_lang] = []
                        if target_lang not in language_texts:
                            language_texts[target_lang] = []
                            
                        language_texts[source_lang].append(source_text)
                        language_texts[target_lang].append(target_text)
        
        # Write monolingual files
        for lang, texts in language_texts.items():
            output_file = processed_dir / f"{lang}_corpus.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                for text in texts:
                    f.write(f"{text}\n")
            self.logger.info(f"Created {output_file} with {len(texts)} sentences")
    
    def create_final_training_file(self):
        """Merge all data into final training file"""
        sampled_dir = Path(self.config['output_dir']) / 'sampled'
        final_dir = Path(self.config['output_dir']) / 'final'
        processed_dir = Path(self.config['output_dir']) / 'processed'
        
        # Collect all files to merge
        files_to_merge = []
        
        # Sampled files
        files_to_merge.extend(sampled_dir.glob('*_sampled.txt'))
        
        # Augmented files
        if final_dir.exists():
            files_to_merge.extend(final_dir.glob('augmented_*.txt'))
            files_to_merge.extend(final_dir.glob('pivot_pairs/*.txt'))
        
        # Merge all
        output_file = processed_dir / 'train_final.txt'
        merge_datasets(files_to_merge, output_file)
        
        self.logger.info(f"Created final training file: {output_file}")

# Update practical_data_pipeline.py
# Add these methods to PracticalDataPipeline class:
def _create_training_ready_data(self):
    """Create data ready for training"""
    connector = PipelineConnector()
    
    # Create monolingual corpora for vocabulary
    self.logger.info("Creating monolingual corpora...")
    connector.create_monolingual_corpora()
    
    # Create final training file
    self.logger.info("Creating final training file...")
    connector.create_final_training_file()