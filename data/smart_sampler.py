# data/smart_sampler.py
import random
import re
from typing import List, Tuple
from pathlib import Path
import mmap
import logging
from tqdm import tqdm

class SmartDataSampler:
    """Sample high-quality data from large corpora with memory efficiency"""
    
    def __init__(self):
        log_dir = Path('log')
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_dir /'data_pipeline.log'), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
        
        self.quality_filters = [
            self.filter_length,
            self.filter_ratio,
            self.filter_numbers,
            self.filter_quality_score
        ]
    
    def sample_high_quality_pairs(self, input_file: str, output_file: str, target_size: int = 1000000) -> None:
        """Sample high-quality sentence pairs with memory-efficient processing"""
        input_file = Path(input_file)
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Count total lines using memory mapping
        with open(input_file, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                total_lines = mm.read().count(b'\n')
        
        quality_lines = []
        
        # First pass: Assess quality
        self.logger.info(f"🔍 Assessing data quality for {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, total=total_lines, desc="Assessing quality")):
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    continue
                source, target = parts
                if self.is_high_quality(source, target):
                    quality_lines.append(i)
        
        self.logger.info(f"✅ Found {len(quality_lines):,} quality pairs out of {total_lines:,}")
        
        # Sample subset if too large
        if len(quality_lines) > target_size:
            sampled_indices = set(random.sample(quality_lines, target_size))
        else:
            sampled_indices = set(quality_lines)
        
        # Second pass: Write sampled data
        self.logger.info(f"💾 Writing sampled data to {output_file}")
        with open(input_file, 'r', encoding='utf-8') as f_in:
            with open(output_file, 'w', encoding='utf-8') as f_out:
                for i, line in enumerate(tqdm(f_in, total=total_lines, desc="Writing samples")):
                    if i in sampled_indices:
                        f_out.write(line)
        
        self.logger.info(f"✅ Wrote {len(sampled_indices):,} sentence pairs")
    
    def is_high_quality(self, source: str, target: str) -> bool:
        """Check if sentence pair is high quality"""
        for filter_func in self.quality_filters:
            if not filter_func(source, target):
                return False
        return True
    
    def filter_length(self, source: str, target: str) -> bool:
        """Filter by length"""
        source_len = len(source.split())
        target_len = len(target.split())
        return 5 <= source_len <= 50 and 5 <= target_len <= 50
    
    def filter_ratio(self, source: str, target: str) -> bool:
        """Filter by length ratio"""
        ratio = len(source) / len(target)
        return 0.5 <= ratio <= 2.0
    
    def filter_numbers(self, source: str, target: str) -> bool:
        """Ensure matching numbers"""
        source_nums = set(re.findall(r'\d+', source))
        target_nums = set(re.findall(r'\d+', target))
        return source_nums == target_nums
    
    def filter_quality_score(self, source: str, target: str) -> bool:
        """Basic quality scoring"""
        if source.count('@') > 2 or target.count('@') > 2:
            return False
        if 'http' in source or 'http' in target:
            return False
        return True