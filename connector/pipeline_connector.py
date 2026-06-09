# connector/pipeline_connector.py
"""Connects data pipeline → vocabulary creation → training"""
from pathlib import Path
import logging
import random
from typing import List, Dict
from tqdm import tqdm
from datetime import datetime

from data.data_utils import merge_datasets
from utils.common_utils import DirectoryManager
from utils.exceptions import DataError
from config.schemas import RootConfig
from utils.constants import DATA_SAMPLED_DIR, DATA_FINAL_DIR, DATA_CORPUS_DIR, TRAIN_FINAL_FILENAME, VAL_FINAL_FILENAME

class PipelineConnector:
    """Connects all pipeline stages"""
    
    def __init__(self, config: RootConfig):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
    def create_monolingual_corpora(self):
        """Split parallel data into monolingual files for vocabulary creation"""
        sampled_dir = Path(self.config.data.processed_dir) / DATA_SAMPLED_DIR
        processed_dir = Path(self.config.data.processed_dir)
        processed_dir.mkdir(exist_ok=True)

        if not sampled_dir.exists():
            self.logger.error(f"Sampled directory not found: {sampled_dir}")
            raise DataError(f"Sampled directory not found: {sampled_dir}")
        
        language_texts = {}
        
        # Read all sampled files with error handling
        for file_path in sampled_dir.glob('*_sampled.txt'):
            try:
                self.logger.info(f"Processing {file_path}")
            
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in tqdm(f, desc=f"Reading {file_path.name}"):
                        parts = line.strip().split('\t')
                        if len(parts) == 4:  # source, target, source_lang, target_lang
                            source_text, target_text, source_lang, target_lang = parts
                            
                            # +++ ADDED: Validate language codes against config +++
                            if source_lang not in self.config.data.active_languages:
                                self.logger.warning(f"Unknown source language: {source_lang}")
                                continue
                            if target_lang not in self.config.data.active_languages:
                                self.logger.warning(f"Unknown target language: {target_lang}")
                                continue

                            # Collect texts by language
                            if source_lang not in language_texts:
                                language_texts[source_lang] = []
                            if target_lang not in language_texts:
                                language_texts[target_lang] = []
                            
                            language_texts[source_lang].append(source_text)
                            language_texts[target_lang].append(target_text)
                        else:
                            self.logger.warning(f"Invalid line format: expected 4 fields, got {len(parts)}")    
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                continue

        # Write monolingual files with deduplication
        corpus_dir = processed_dir / DATA_CORPUS_DIR
        corpus_dir.mkdir(parents=True, exist_ok=True)
        for lang, texts in language_texts.items():
            output_file = corpus_dir / f"{lang}_corpus.txt"
            # Deduplicate texts
            unique_texts = list(dict.fromkeys(texts))  # Preserves order

            with open(output_file, 'w', encoding='utf-8') as f:
                for text in unique_texts:
                    f.write(f"{text}\n")
            self.logger.info(f"Created {output_file} with {len(unique_texts)} unique sentences (from {len(texts)} total)")
    
    def create_final_training_file(self):
        """Merge all data into final training file"""
        sampled_dir = Path(self.config.data.processed_dir) / DATA_SAMPLED_DIR
        final_dir = Path(self.config.data.processed_dir) / DATA_FINAL_DIR
        processed_dir = Path(self.config.data.processed_dir)
        
        # Collect all files to merge
        files_to_merge = []
        
        # Sampled files
        files_to_merge.extend(sampled_dir.glob('*_sampled.txt'))
        
        # Augmented files
        if final_dir.exists():
            files_to_merge.extend(final_dir.glob('augmented_*.txt'))
            files_to_merge.extend(final_dir.glob('pivot_pairs/*.txt'))
        
        # Merge all
        output_file = processed_dir / TRAIN_FINAL_FILENAME
        merge_datasets(files_to_merge, output_file)
        
        # Split into train and validation sets
        val_file = processed_dir / VAL_FINAL_FILENAME
        temp_file = processed_dir / 'train_temp.txt'
        with open(output_file, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
        random.shuffle(lines)
        split_idx = int(len(lines) * 0.9)
        with open(temp_file, 'w', encoding='utf-8') as f_train:
            f_train.writelines(lines[:split_idx])
        with open(val_file, 'w', encoding='utf-8') as f_val:
            f_val.writelines(lines[split_idx:])
        temp_file.replace(output_file)
        
        self.logger.info(f"Created final training file: {output_file} ({split_idx} lines)")
        self.logger.info(f"Created validation file: {val_file} ({len(lines) - split_idx} lines)")

    def get_data_version_info(self) -> Dict[str, str]:
        """Get data and pipeline version information"""
        return {
            'data_version': getattr(self.config, 'version', '1.0.0'),
            'timestamp': datetime.now().isoformat(),
            'git_commit': self._get_git_commit()
        }

    def _get_git_commit(self) -> str:
        """Get current git commit hash"""
        try:
            import subprocess
            return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
        except Exception:
            return "unknown"
