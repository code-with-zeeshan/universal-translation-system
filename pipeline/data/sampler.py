# pipeline/data/sampler.py
"""
Smart Data Sampler with optional LaBSE GPU embedding filter.
"""

import random
import re
import mmap
from pathlib import Path
from typing import List, Tuple, Callable, Dict, Optional
from tqdm import tqdm
import asyncio
import aiofiles
import itertools
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import numpy as np

import logging

from utils.common_utils import DirectoryManager
from utils.gpu_utils import get_gpu_profile, torch_is_available

try:
    from sentence_transformers import SentenceTransformer, util
except Exception:
    SentenceTransformer = None


class SmartDataSampler:
    """Sample high-quality data from large corpora with memory efficiency and async support"""
    
    def __init__(self, log_dir: str = None, use_gpu_embedding: bool = True):
        """
        Initialize sampler with standardized logging
        
        Args:
            log_dir: Custom log directory (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.use_gpu_embedding = use_gpu_embedding and SentenceTransformer is not None and torch_is_available()
        
        self.quality_filters: List[Callable[[str, str], bool]] = [
            self.filter_length,
            self.filter_ratio,
            self.filter_numbers,
            self.filter_quality_score,
        ]
        
        self._sentence_model = None
        self._embed_cache: Dict[str, np.ndarray] = {}
        
        if self.use_gpu_embedding:
            gpu = get_gpu_profile()
            self.embed_batch_size = gpu.labse_batch_size
            self.logger.info(f"📊 LaBSE GPU embedding filter enabled (batch_size={self.embed_batch_size}, profile={gpu.name})")
        else:
            self.embed_batch_size = 1
        
        self.logger.info(f"📊 Initialized SmartDataSampler with {len(self.quality_filters)} quality filters")
    
    @property
    def sentence_model(self):
        if self._sentence_model is None and self.use_gpu_embedding:
            self.logger.info("Loading LaBSE sentence transformer for cross-lingual quality...")
            self._sentence_model = SentenceTransformer('sentence-transformers/LaBSE')
            if torch_is_available():
                self._sentence_model = self._sentence_model.to('cuda')
        return self._sentence_model
    
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts with LaBSE on GPU."""
        model = self.sentence_model
        if model is None:
            return np.array([])
        return model.encode(texts, batch_size=self.embed_batch_size, convert_to_numpy=True, show_progress_bar=False)
    
    def _cosine_similarity_filter(self, sources: List[str], targets: List[str], threshold: float = 0.5) -> List[bool]:
        """Filter pairs by cross-lingual embedding similarity.
        
        Returns a boolean mask: True if similarity >= threshold.
        """
        if not self.use_gpu_embedding or not sources:
            return [True] * len(sources)
        
        src_emb = self._embed_texts(sources)
        tgt_emb = self._embed_texts(targets)
        if src_emb.size == 0 or tgt_emb.size == 0:
            return [True] * len(sources)
        
        sims = util.cos_sim(src_emb, tgt_emb).diagonal().cpu().numpy()
        return [s >= threshold for s in sims]

    def _process_file_in_batches(self, file_path: Path, batch_size: int = 10000):
        """Process file in batches for better memory efficiency"""
        with open(file_path, 'r', encoding='utf-8') as f:
            while True:
                lines = list(itertools.islice(f, batch_size))
                if not lines:
                    break
                yield lines  
    
    async def sample_high_quality_pairs_async(
        self, 
        input_file: str, 
        output_file: str, 
        target_size: int = 1000000,
        source_lang: str = None,  
        target_lang: str = None,
        chunk_size: int = 10000  
    ) -> dict:
        """
        Async version of sampling for better performance
        with high-quality sentence pairs and memory-efficient processing
        
        Args:
            input_file: Path to input file
            output_file: Path to output file
            target_size: Target number of sentences
            source_lang: Source language code
            target_lang: Target language code
            chunk_size: Size of chunks to process
            
        Returns:
            Dictionary with sampling statistics
        """
        input_path = Path(input_file)
        output_path = Path(output_file)
        
        # Use centralized directory creation
        DirectoryManager.create_directory(output_path.parent)
        
        # Standardized logging messages
        self.logger.info("🔍 ASYNC SAMPLING PROCESS STARTED")
        self.logger.info(f"📁 Input file: {input_path}")
        self.logger.info(f"📁 Output file: {output_path}")
        self.logger.info(f"🎯 Target size: {target_size:,} sentences")
        
        # Count total lines asynchronously using memory mapping
        total_lines = await self._count_lines_async(input_path)
        self.logger.info(f"📊 Total lines in input: {total_lines:,}")
        
        # Process in chunks using process pool
        with ProcessPoolExecutor() as executor:
            # First pass: Quality assessment in parallel
            quality_lines = await self._assess_quality_async(
                input_path, total_lines, executor, chunk_size
            )
        
            # Calculate statistics
            quality_percentage = (len(quality_lines) / total_lines * 100) if total_lines > 0 else 0
            self.logger.info(f"✅ Quality assessment complete: {len(quality_lines):,} quality pairs ({quality_percentage:.1f}%)")

            # Optional: LaBSE embedding similarity filter (GPU)
            if self.use_gpu_embedding:
                self.logger.info(f"🔬 Running LaBSE embedding similarity filter (batch_size={self.embed_batch_size})...")
                quality_lines = await self._embedding_filter_pass_async(
                    input_path, quality_lines, source_lang, target_lang
                )
                self.logger.info(f"✅ After embedding filter: {len(quality_lines):,} quality pairs remain")

            # Sample subset if needed
            sampled_indices = self._sample_indices(quality_lines, target_size)

            # Second pass: Write sampled data asynchronously
            written_count = await self._write_sampled_data_async(
                input_path, output_path, sampled_indices, total_lines, 
                source_lang, target_lang
            )
        
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
        self.logger.info("✅ ASYNC SAMPLING PROCESS COMPLETED")
        
        return stats

    def sample_high_quality_pairs(
        self, 
        input_file: str, 
        output_file: str, 
        target_size: int = 1000000,
        source_lang: str = None,  
        target_lang: str = None
    ) -> dict:
        """
        Synchronous version of sampling with batch processing
    
        Args:
            input_file: Path to input file
            output_file: Path to output file
            target_size: Target number of sentences
            source_lang: Source language code
            target_lang: Target language code
        
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
        total_lines = self._count_lines_mmap(input_path)
        self.logger.info(f"📊 Total lines in input: {total_lines:,}")
    
        # First pass: Quality assessment with batch processing
        quality_lines = []
        line_idx = 0
    
        # Process file in batches for better memory efficiency
        for batch in self._process_file_in_batches(input_path, batch_size=10000):
            for line in batch:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    source, target = parts[0], parts[1]
                    if self.is_high_quality(source, target):
                        quality_lines.append(line_idx)
                line_idx += 1
        
            # Log progress
            if line_idx % 100000 == 0:
                self.logger.info(f"Processed {line_idx:,}/{total_lines:,} lines...")
    
        # Calculate statistics
        quality_percentage = (len(quality_lines) / total_lines * 100) if total_lines > 0 else 0
        self.logger.info(f"✅ Quality assessment complete: {len(quality_lines):,} quality pairs ({quality_percentage:.1f}%)")
    
        # Sample subset if needed
        sampled_indices = self._sample_indices(quality_lines, target_size)
    
        # Second pass: Write sampled data with batch processing
        written_count = 0
        line_idx = 0
    
        self.logger.info("💾 Writing sampled data...")
    
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for batch in self._process_file_in_batches(input_path, batch_size=10000):
                batch_start_idx = line_idx
                for i, line in enumerate(batch):
                    current_idx = batch_start_idx + i
                    if current_idx in sampled_indices:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            source, target = parts[0], parts[1]
                            f_out.write(f"{source}\t{target}\t{source_lang}\t{target_lang}\n")
                            written_count += 1
                line_idx += len(batch)
    
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
    
    async def _count_lines_async(self, file_path: Path) -> int:
        """Count lines asynchronously"""
        line_count = 0
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            async for _ in f:
                line_count += 1
        return line_count

    def _count_lines_mmap(self, file_path: Path) -> int:
        """Count lines using memory mapping for efficiency"""
        with open(file_path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                lines = 0
                while mmapped_file.readline():
                    lines += 1
                return lines
    
    async def _assess_quality_async(self, input_path: Path, total_lines: int,executor: ProcessPoolExecutor, chunk_size: int) -> List[int]:
        """Assess quality asynchronously using process pool"""
        quality_lines = []
        
        self.logger.info("🔍 Starting quality assessment...")
        
        # Read file in chunks and process in parallel
        chunks = []
        async with aiofiles.open(input_path, 'r', encoding='utf-8') as f:
            chunk = []
            line_idx = 0
            
            async for line in f:
                chunk.append((line_idx, line))
                line_idx += 1
                
                if len(chunk) >= chunk_size:
                    chunks.append(chunk)
                    chunk = []
            
            if chunk:
                chunks.append(chunk)
        
        # Process chunks in parallel
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(executor, self._process_chunk, chunk)
            for chunk in chunks
        ]
        
        results = await asyncio.gather(*futures)
        
        # Combine results
        for chunk_quality_indices in results:
            quality_lines.extend(chunk_quality_indices)
        
        return quality_lines

    def _process_chunk(self, chunk: List[Tuple[int, str]]) -> List[int]:
        """Process a chunk of lines (runs in process pool)"""
        quality_indices = []
        
        for idx, line in chunk:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                source, target = parts
                if self.is_high_quality(source, target):
                    quality_indices.append(idx)
        
        return quality_indices
    
    def _sample_indices(self, quality_lines: List[int], target_size: int) -> set:
        """Sample indices from quality lines"""
        if len(quality_lines) > target_size:
            sampled = set(random.sample(quality_lines, target_size))
            self.logger.info(f"🎲 Randomly sampled {target_size:,} from {len(quality_lines):,} quality lines")
        else:
            sampled = set(quality_lines)
            self.logger.info(f"📋 Using all {len(quality_lines):,} quality lines (below target)")
        
        return sampled
    
    async def _write_sampled_data_async(self, input_path: Path, output_path: Path, sampled_indices: set, total_lines: int,source_lang: str, target_lang: str) -> int:
        """Write sampled data asynchronously to output file with language codes"""
        written_count = 0
        line_idx = 0
        
        self.logger.info("💾 Writing sampled data asynchronously...")
        
        async with aiofiles.open(input_path, 'r', encoding='utf-8') as f_in:
            async with aiofiles.open(output_path, 'w', encoding='utf-8') as f_out:
                async for line in f_in:
                    if line_idx in sampled_indices:
                        parts = line.strip().split('\t')
                        if len(parts) == 2:
                            source, target = parts
                            await f_out.write(f"{source}\t{target}\t{source_lang}\t{target_lang}\n")
                            written_count += 1
                    
                    line_idx += 1
                    
                    # Progress update every 10000 lines
                    if line_idx % 10000 == 0:
                        self.logger.debug(f"Processed {line_idx:,}/{total_lines:,} lines")
        
        return written_count
    
    async def _embedding_filter_pass_async(
        self, input_path: Path, quality_indices: List[int],
        source_lang: Optional[str], target_lang: Optional[str],
        threshold: float = 0.5,
    ) -> List[int]:
        """Filter quality_indices by LaBSE cross-lingual similarity on GPU.
        
        Reads lines for quality_indices in batches, computes embedding
        similarity, and keeps only pairs above threshold.
        """
        if not self.use_gpu_embedding or not quality_indices:
            return quality_indices
        
        idx_set = set(quality_indices)
        batch_sources: List[str] = []
        batch_targets: List[str] = []
        batch_indices: List[int] = []
        kept: List[int] = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                if line_idx not in idx_set:
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    batch_sources.append(parts[0])
                    batch_targets.append(parts[1])
                    batch_indices.append(line_idx)
                
                if len(batch_sources) >= self.embed_batch_size:
                    mask = self._cosine_similarity_filter(batch_sources, batch_targets, threshold)
                    for i, ok in enumerate(mask):
                        if ok:
                            kept.append(batch_indices[i])
                    batch_sources, batch_targets, batch_indices = [], [], []
        
        # Final batch
        if batch_sources:
            mask = self._cosine_similarity_filter(batch_sources, batch_targets, threshold)
            for i, ok in enumerate(mask):
                if ok:
                    kept.append(batch_indices[i])
        
        return kept
    
    @staticmethod
    def _has_no_word_boundaries(text: str) -> bool:
        """Check if text uses a script without word-boundary spaces (CJK, Thai, Lao, etc.)."""
        return bool(re.search(
            r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff'  # CJK Unified
            r'\uac00-\ud7af'                              # Korean Hangul
            r'\u0e00-\u0e7f'                              # Thai
            r'\u0e80-\u0eff'                              # Lao
            r'\u1000-\u109f'                              # Myanmar (Burmese)
            r'\u1780-\u17ff'                              # Khmer
            r'\u0f00-\u0fff'                              # Tibetan
            r'\u1200-\u137f'                              # Ethiopic (Amharic, Ge'ez)
            r'\u1c00-\u1c4f'                              # Lepcha
            r'\ua000-\ua4cf'                              # Yi
            r'\u1400-\u167f'                              # Unified Canadian Aboriginal
            r'\u2c00-\u2c5f'                              # Glagolitic
            r'\u0370-\u03ff\u1f00-\u1fff'                 # Greek (polytonic, no spaces)
            r'\u0590-\u05ff'                              # Hebrew
            r'\u0600-\u06ff'                              # Arabic (cursive, variable width)
            r'\u0900-\u097f'                              # Devanagari (Hindi, Sanskrit)
            r'\u0980-\u09ff'                              # Bengali
            r'\u0a00-\u0a7f'                              # Gurmukhi
            r'\u0a80-\u0aff'                              # Gujarati
            r'\u0b00-\u0b7f'                              # Oriya
            r'\u0b80-\u0bff'                              # Tamil
            r'\u0c00-\u0c7f'                              # Telugu
            r'\u0c80-\u0cff'                              # Kannada
            r'\u0d00-\u0d7f'                              # Malayalam
            r'\u0d80-\u0dff'                              # Sinhala
            r'\u0e00-\u0e7f'                              # Thai (dup for clarity)
            r']', text))

    @staticmethod
    def _script_aware_len(text: str) -> int:
        """Word count for Latin-like scripts, character count for scripts lacking word boundaries."""
        if SmartDataSampler._has_no_word_boundaries(text):
            return len(text)
        words = text.split()
        return len(words) if words else len(text)

    def is_high_quality(self, source: str, target: str) -> bool:
        """Check if sentence pair meets all quality criteria"""
        for filter_func in self.quality_filters:
            if not filter_func(source, target):
                return False
        return True
    
    def filter_length(self, source: str, target: str) -> bool:
        """Filter by word/character count length (script-aware for CJK/Thai)."""
        source_len = self._script_aware_len(source)
        target_len = self._script_aware_len(target)
        return 5 <= source_len <= 50 and 5 <= target_len <= 50
    
    def filter_ratio(self, source: str, target: str) -> bool:
        """Filter by character length ratio (looser bounds for non-Latin scripts)."""
        if len(target) == 0:
            return False
        ratio = len(source) / len(target)
        if self._has_no_word_boundaries(source) or self._has_no_word_boundaries(target):
            return 0.25 <= ratio <= 4.0
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

    def calculate_quality_metrics(self, file_path: str) -> Dict[str, float]:
        """Calculate quality metrics for a dataset"""
        metrics = {
            'avg_sentence_length': 0,
            'length_ratio_variance': 0,
            'unique_sentence_ratio': 0,
            'quality_score': 0
        }
        # Implementation here
        return metrics      

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
