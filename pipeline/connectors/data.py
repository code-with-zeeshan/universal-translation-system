# pipeline/connectors/data.py
"""Connects data pipeline → vocabulary creation → training"""
from pathlib import Path
import logging
import random
from typing import List, Dict
from tqdm import tqdm
from datetime import datetime

from pipeline.data.utils import merge_datasets
from utils.common_utils import DirectoryManager, RuntimeDirectoryManager
from utils.exceptions import DataError
from config.schemas import RootConfig
from utils.constants import TRAIN_FINAL_FILENAME, VAL_FINAL_FILENAME

class PipelineConnector:
    """Connects all pipeline stages"""
    
    def __init__(self, config: RootConfig):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.runtime_dirs = RuntimeDirectoryManager()
        
    def create_monolingual_corpora(self):
        """Split parallel data into monolingual files for vocabulary creation"""
        sampled_dir = self.runtime_dirs.sampled_dir
        processed_dir = self.runtime_dirs.processed_dir

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

        # Also extract from augmented data (now in 4-column format)
        final_dir = self.runtime_dirs.augment_dir
        if final_dir.exists():
            aug_files = list(final_dir.glob('*.txt')) + list(final_dir.glob('*/*.txt'))
            for file_path in aug_files:
                try:
                    self.logger.info(f"Processing augmented {file_path}")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split('\t')
                            if len(parts) == 4:
                                _, _, src_lang, tgt_lang = parts
                                if src_lang not in language_texts:
                                    language_texts[src_lang] = []
                                if tgt_lang not in language_texts:
                                    language_texts[tgt_lang] = []
                except Exception as e:
                    self.logger.warning(f"Skipping {file_path}: {e}")
                    continue

        # Write monolingual files with deduplication
        corpus_dir = self.runtime_dirs.corpus_dir
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
        sampled_dir = self.runtime_dirs.sampled_dir
        final_dir = self.runtime_dirs.augment_dir
        processed_dir = self.runtime_dirs.processed_dir
        
        # Collect all files to merge
        files_to_merge = []

        # Sampled files
        files_to_merge.extend(sampled_dir.glob('*_sampled.txt'))

        # Augmented files (backtranslations, idioms, false friends, etc.)
        if final_dir.exists():
            files_to_merge.extend(final_dir.glob('*.txt'))
            files_to_merge.extend(final_dir.glob('*/*.txt'))
            files_to_merge.extend(final_dir.glob('pivot_pairs/*.txt'))
        
        # Merge all
        output_file = self.runtime_dirs.train_final_path
        merge_datasets(files_to_merge, output_file)
        
        # Split into train and validation sets
        val_file = self.runtime_dirs.val_final_path
        temp_file = self.runtime_dirs.train_temp_path
        with open(output_file, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
        rng = random.Random(self.config.data.seed)
        rng.shuffle(lines)
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
