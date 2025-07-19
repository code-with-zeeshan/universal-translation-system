# data/smart_sampler.py
"""
Smart Data Sampler - Refactored with Low Priority Clean-up
- Standardized logging
- Centralized directory creation
- Cleaned imports
- Standardized logging messages
"""

import random
import re
import mmap
from pathlib import Path
from typing import List, Tuple, Callable
from tqdm import tqdm

# Clean import from utils
from utils.common_utils import DirectoryManager, StandardLogger

class SmartDataSampler:
    """Sample high-quality data from large corpora with memory efficiency"""
    
    def __init__(self, log_dir: str = None):
        """
        Initialize sampler with standardized logging
        
        Args:
            log_dir: Custom log directory (optional)
        """
        self.logger = StandardLogger.get_logger(__name__, log_dir)
        StandardLogger.log_system_info(self.logger)
        
        # Initialize quality filters
        self.quality_filters: List[Callable[[str, str], bool]] = [
            self.filter_length,
            self.filter_ratio,
            self.filter_numbers,
            self.filter_quality_score
        ]
        
        self.logger.info(f"📊 Initialized SmartDataSampler with {len(self.quality_filters)} quality filters")
    
    def sample_high_quality_pairs(
        self, 
        input_file: str, 
        output_file: str, 
        target_size: int = 1000000,
        source_lang: str = None,  
        target_lang: str = None   
    ) -> dict:
        """
        Sample high-quality sentence pairs with memory-efficient processing
        
        Args:
            input_file: Path to input file
            output_file: Path to output file
            target_size: Target number of sentences
            
        Returns:
            Dictionary with sampling statistics
        """
        input_path = Path(input_file)
        output_path = Path(output_file)
        
        # Use centralized directory creation
        DirectoryManager.create_directory(output_path.parent)
        
        # Standardized logging messages
        self.logger.info("🔍 SAMPLING PROCESS STARTED")
        self.logger.info(f"📁 Input file: {input_path}")
        self.logger.info(f"📁 Output file: {output_path}")
        self.logger.info(f"🎯 Target size: {target_size:,} sentences")
        
        # Count total lines using memory mapping
        total_lines = self._count_lines_efficiently(input_path)
        self.logger.info(f"📊 Total lines in input: {total_lines:,}")
        
        # First pass: Quality assessment
        quality_lines = self._assess_quality(input_path, total_lines)
        
        # Calculate statistics
        quality_percentage = (len(quality_lines) / total_lines * 100) if total_lines > 0 else 0
        self.logger.info(f"✅ Quality assessment complete: {len(quality_lines):,} quality pairs ({quality_percentage:.1f}%)")
        
        # Sample subset if needed
        sampled_indices = self._sample_indices(quality_lines, target_size)
        
        # Second pass: Write sampled data
        written_count = self._write_sampled_data(input_path, output_path, sampled_indices, total_lines, source_lang, target_lang)
        
        # Final statistics
        stats = {
            'total_lines': total_lines,
            'quality_lines': len(quality_lines),
            'quality_percentage': quality_percentage,
            'target_size': target_size,
            'written_count': written_count,
            'sampling_ratio': (written_count / total_lines * 100) if total_lines > 0 else 0
        }
        
        self.logger.info("📈 SAMPLING STATISTICS")
        self.logger.info(f"  📊 Total lines processed: {stats['total_lines']:,}")
        self.logger.info(f"  ✅ Quality lines found: {stats['quality_lines']:,} ({stats['quality_percentage']:.1f}%)")
        self.logger.info(f"  📝 Lines written: {stats['written_count']:,}")
        self.logger.info(f"  📉 Overall sampling ratio: {stats['sampling_ratio']:.1f}%")
        self.logger.info("✅ SAMPLING PROCESS COMPLETED")
        
        return stats
    
    def _count_lines_efficiently(self, file_path: Path) -> int:
        """Count lines using memory mapping for efficiency"""
        try:
            with open(file_path, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    return mm.read().count(b'\n')
        except Exception as e:
            self.logger.error(f"❌ Failed to count lines in {file_path}: {e}")
            return 0
    
    def _assess_quality(self, input_path: Path, total_lines: int) -> List[int]:
        """Assess quality of all lines and return indices of quality pairs"""
        quality_lines = []
        
        self.logger.info("🔍 Starting quality assessment...")
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(tqdm(f, total=total_lines, desc="Quality assessment")):
                    parts = line.strip().split('\t')
                    if len(parts) != 2:
                        continue
                    
                    source, target = parts
                    if self.is_high_quality(source, target):
                        quality_lines.append(i)
        except Exception as e:
            self.logger.error(f"❌ Quality assessment failed: {e}")
            raise
        
        return quality_lines
    
    def _sample_indices(self, quality_lines: List[int], target_size: int) -> set:
        """Sample indices from quality lines"""
        if len(quality_lines) > target_size:
            sampled = set(random.sample(quality_lines, target_size))
            self.logger.info(f"🎲 Randomly sampled {target_size:,} from {len(quality_lines):,} quality lines")
        else:
            sampled = set(quality_lines)
            self.logger.info(f"📋 Using all {len(quality_lines):,} quality lines (below target)")
        
        return sampled
    
    def _write_sampled_data(self, input_path: Path, output_path: Path, sampled_indices: set, total_lines: int,source_lang: str, target_lang: str) -> int:
        """Write sampled data to output file with language codes"""
        written_count = 0
        
        self.logger.info("💾 Writing sampled data...")
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f_in:
                with open(output_path, 'w', encoding='utf-8') as f_out:
                    for i, line in enumerate(tqdm(f_in, total=total_lines, desc="Writing samples")):
                        if i in sampled_indices:
                            source, target = line.strip().split('\t')
                            f_out.write(f"{source}\t{target}\t{source_lang}\t{target_lang}\n")
                            written_count += 1
        except Exception as e:
            self.logger.error(f"❌ Failed to write sampled data: {e}")
            raise
        
        return written_count
    
    def is_high_quality(self, source: str, target: str) -> bool:
        """Check if sentence pair meets all quality criteria"""
        for filter_func in self.quality_filters:
            if not filter_func(source, target):
                return False
        return True
    
    def filter_length(self, source: str, target: str) -> bool:
        """Filter by word count length"""
        source_len = len(source.split())
        target_len = len(target.split())
        return 5 <= source_len <= 50 and 5 <= target_len <= 50
    
    def filter_ratio(self, source: str, target: str) -> bool:
        """Filter by character length ratio"""
        if len(target) == 0:
            return False
        ratio = len(source) / len(target)
        return 0.5 <= ratio <= 2.0
    
    def filter_numbers(self, source: str, target: str) -> bool:
        """Ensure numeric consistency between source and target"""
        source_nums = set(re.findall(r'\d+', source))
        target_nums = set(re.findall(r'\d+', target))
        return source_nums == target_nums
    
    def filter_quality_score(self, source: str, target: str) -> bool:
        """Apply basic quality heuristics"""
        # Too many @ symbols (likely metadata)
        if source.count('@') > 2 or target.count('@') > 2:
            return False
        
        # Contains URLs
        if 'http' in source.lower() or 'http' in target.lower():
            return False
        
        # Too many repeated characters
        if re.search(r'(.)\1{4,}', source) or re.search(r'(.)\1{4,}', target):
            return False
        
        # Very short sentences
        if len(source.strip()) < 10 or len(target.strip()) < 10:
            return False
        
        return True

# Example usage
if __name__ == "__main__":
    # Test the refactored sampler
    sampler = SmartDataSampler()
    
    # Create test data
    test_input = Path("test_data/sample_pairs.txt")
    DirectoryManager.create_directory(test_input.parent)
    
    # Generate sample data for testing
    with open(test_input, 'w', encoding='utf-8') as f:
        f.write("Hello world\tHola mundo\n")
        f.write("This is a test\tEsta es una prueba\n")
        f.write("Good morning\tBuenos días\n")
        f.write("@@@@@\t@@@@@\n")  # Should be filtered out
        f.write("http://example.com\thttp://ejemplo.com\n")  # Should be filtered out
    
    # Test sampling
    stats = sampler.sample_high_quality_pairs(
        input_file=str(test_input),
        output_file="test_data/sampled_pairs.txt",
        target_size=2
    )
    
    print(f"Sampling completed with stats: {stats}")