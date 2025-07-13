# data/smart_sampler.py
import random
from typing import List, Tuple

class SmartDataSampler:
    """Sample high-quality data from large corpora"""
    
    def __init__(self):
        self.quality_filters = [
            self.filter_length,
            self.filter_ratio,
            self.filter_numbers,
            self.filter_quality_score
        ]
    
    def sample_high_quality_pairs(
        self, 
        input_file: str, 
        output_file: str,
        target_size: int = 1000000  # 1M sentences
    ):
        """Sample only high-quality sentence pairs"""
        
        # First pass: count total lines and assess quality
        print("🔍 First pass: Assessing data quality...")
        total_lines = 0
        quality_lines = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines += 1
                
                if total_lines % 100000 == 0:
                    print(f"  Processed {total_lines:,} lines...")
                
                # Parse parallel sentence
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    continue
                    
                source, target = parts
                
                # Apply quality filters
                if self.is_high_quality(source, target):
                    quality_lines.append(i)
        
        print(f"✅ Found {len(quality_lines):,} quality pairs out of {total_lines:,}")
        
        # Sample subset if too large
        if len(quality_lines) > target_size:
            sampled_indices = set(random.sample(quality_lines, target_size))
        else:
            sampled_indices = set(quality_lines)
        
        # Second pass: Write sampled data
        print("💾 Second pass: Writing sampled data...")
        with open(input_file, 'r', encoding='utf-8') as f_in:
            with open(output_file, 'w', encoding='utf-8') as f_out:
                for i, line in enumerate(f_in):
                    if i in sampled_indices:
                        f_out.write(line)
        
        print(f"✅ Wrote {len(sampled_indices):,} sentence pairs")
    
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
        
        # Good length: 5-50 words
        if source_len < 5 or source_len > 50:
            return False
        if target_len < 5 or target_len > 50:
            return False
            
        return True
    
    def filter_ratio(self, source: str, target: str) -> bool:
        """Filter by length ratio"""
        ratio = len(source) / len(target)
        
        # Good ratio: 0.5 - 2.0
        return 0.5 <= ratio <= 2.0
    
    def filter_numbers(self, source: str, target: str) -> bool:
        """Ensure matching numbers"""
        source_nums = set(re.findall(r'\d+', source))
        target_nums = set(re.findall(r'\d+', target))
        
        # Numbers should match
        return source_nums == target_nums
    
    def filter_quality_score(self, source: str, target: str) -> bool:
        """Basic quality scoring"""
        # Filter out pairs with too many special characters
        if source.count('@') > 2 or target.count('@') > 2:
            return False
            
        # Filter out pairs with too many URLs
        if 'http' in source or 'http' in target:
            return False
            
        return True