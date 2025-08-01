# tools/create_vocabulary_packs.py
"""
Advanced vocabulary pack creator with corpus analysis and optimization.

This module provides tools for creating optimized vocabulary packs for different 
language groups in NLP applications. It focuses on corpus analysis, intelligent 
vocabulary selection, and compression optimization.

Usage:
    creator = VocabularyPackCreator(corpus_paths)
    pack = creator.create_pack(['en', 'es', 'fr'], 'latin_optimized')
"""

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential

import msgpack
import numpy as np
import sentencepiece as spm
import tempfile
# from transformers import AutoTokenizer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class VocabConfig:
    """Configuration for vocabulary creation."""
    target_size: int = 25000
    character_coverage: float = 0.9995
    model_type: str = 'bpe'
    num_threads: int = 16
    min_token_frequency: int = 10
    subword_ratio: float = 0.3  # 30% of vocab for subwords
    compression_level: int = 9


@dataclass
class VocabStats:
    """Statistics for vocabulary coverage and performance."""
    total_tokens: int
    coverage_percentage: float
    size_mb: float
    compression_ratio: float
    oov_rate: float  # Out-of-vocabulary rate


class VocabularyPackCreator:
    """
    Advanced vocabulary pack creator with corpus analysis and optimization.
    
    This class provides sophisticated tools for creating optimized vocabulary packs
    by analyzing token frequencies, selecting optimal vocabularies, and optimizing
    for compression and coverage.
    
    Attributes:
        corpus_paths: Dictionary mapping language codes to corpus file paths
        config: Configuration object with vocabulary creation parameters
        tokenizer: SentencePiece processor for tokenization
    """
    
    def __init__(
        self, 
        corpus_paths: Dict[str, str], 
        config: Optional[VocabConfig] = None
    ) -> None:
        """
        Initialize the vocabulary pack creator.
        
        Args:
            corpus_paths: Dictionary mapping language codes to corpus file paths
            config: Configuration object (uses default if None)
            
        Raises:
            FileNotFoundError: If any corpus file doesn't exist
            ValueError: If corpus_paths is empty
        """
        if not corpus_paths:
            raise ValueError("corpus_paths cannot be empty")
            
        self.corpus_paths = corpus_paths
        self.config = config or VocabConfig()
        self.tokenizer = spm.SentencePieceProcessor()
        
        # Validate corpus files exist
        self._validate_corpus_files()
        
        logger.info(f"Initialized VocabularyPackCreator with {len(corpus_paths)} languages")

    def _calculate_quality_metrics(self, vocab: Dict[str, int], corpus_path: str) -> Dict[str, float]:
        """Calculate comprehensive quality metrics for vocabulary"""
        from collections import defaultdict
        import numpy as np
        # Use chunked reading for large files
        from itertools import islice
    
        metrics = {
            'unigram_coverage': 0.0,
            'bigram_coverage': 0.0,
            'fertility': 0.0,  # Average tokens per word
            'ambiguity': 0.0,  # Average words per token
            'compression_rate': 0.0,
            'oov_rate': 0.0
        }
    
        # Use reservoir sampling for large corpora
        sample_size = min(10000, self._count_lines(corpus_path))
        
        # Sample corpus for metrics
        token_to_words = defaultdict(set)
        word_to_tokens = defaultdict(list)
        total_words = 0
        oov_words = 0
        covered_chars = 0
        total_chars = 0
    
        logger.info("Calculating vocabulary quality metrics...")
    
        try:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= sample_size:
                        break
                
                    words = line.strip().split()
                    total_words += len(words)
                
                    for word in words:
                        total_chars += len(word)
                    
                        # Try to tokenize the word
                        tokens = self._tokenize_word(word, vocab)
                    
                        if tokens == ['<unk>']:
                            oov_words += 1
                        else:
                            covered_chars += len(word)
                            word_to_tokens[word] = tokens
                        
                            for token in tokens:
                                token_to_words[token].add(word)
        
            # Calculate metrics
            if total_words > 0:
                metrics['oov_rate'] = oov_words / total_words
                metrics['unigram_coverage'] = 1.0 - metrics['oov_rate']
        
            if total_chars > 0:
                metrics['compression_rate'] = covered_chars / total_chars
        
            # Fertility: average tokens per word
            if word_to_tokens:
                fertilities = [len(tokens) for tokens in word_to_tokens.values()]
                metrics['fertility'] = np.mean(fertilities)
        
            # Ambiguity: average words per token
            if token_to_words:
                ambiguities = [len(words) for words in token_to_words.values()]
                metrics['ambiguity'] = np.mean(ambiguities)
        
            # Estimate bigram coverage (simplified)
            bigram_coverage_estimate = metrics['unigram_coverage'] ** 2
            metrics['bigram_coverage'] = bigram_coverage_estimate
        
            logger.info("Quality metrics calculated:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
        
        except Exception as e:
            logger.error(f"Failed to calculate quality metrics: {e}")
    
        return metrics

    def _count_lines(self, file_path: Union[str, Path]) -> int:
        """Count lines in a file efficiently"""
        count = 0
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
           for _ in f:
               count += 1
        return count    

    def _tokenize_word(self, word: str, vocab: Dict[str, int]) -> List[str]:
        """Tokenize a word using the vocabulary (simple greedy approach)"""
        if word in vocab:
            return [word]
    
        # Try subword tokenization
        tokens = []
        i = 0
    
        while i < len(word):
            # Find longest matching prefix
            longest_match = None
            for j in range(len(word), i, -1):
                subword = word[i:j]
            
                # Add ## prefix for continuation subwords
                if i > 0:
                    subword = f"##{subword}"
            
                if subword in vocab:
                    longest_match = subword
                    i = j
                    break
        
            if longest_match:
                tokens.append(longest_match)
            else:
                # No match found, skip character
                i += 1
    
        return tokens if tokens else ['<unk>']

    def analyze_vocabulary_quality(self, pack_name: str, test_corpus: Optional[str] = None) -> Dict[str, Any]:
        """Analyze the quality of a created vocabulary pack"""
    
        # Load the pack
        pack_path = Path('vocabs') / f'{pack_name}_v1.json'
        if not pack_path.exists():
            raise FileNotFoundError(f"Pack not found: {pack_path}")
    
        with open(pack_path, 'r') as f:
            pack = json.load(f)
    
        # Combine all vocabularies
        all_vocab = {}
        all_vocab.update(pack['tokens'])
        all_vocab.update(pack['subwords'])
        all_vocab.update(pack['special_tokens'])
    
        # Use test corpus or original corpus
        if test_corpus:
            corpus_path = test_corpus
        else:
            # Try to find a corpus for one of the pack's languages
            corpus_path = None
            for lang in pack['languages']:
                potential_path = self.corpus_paths.get(lang)
                if potential_path and Path(potential_path).exists():
                    corpus_path = potential_path
                    break
    
        if not corpus_path:
            logger.warning("No corpus found for quality analysis")
            return {}
    
        # Calculate metrics
        metrics = self._calculate_quality_metrics(all_vocab, corpus_path)
    
        # Add pack metadata
        analysis = {
            'pack_name': pack_name,
            'languages': pack['languages'],
            'vocabulary_size': len(all_vocab),
            'token_distribution': {
                'tokens': len(pack['tokens']),
                'subwords': len(pack['subwords']),
                'special_tokens': len(pack['special_tokens'])
            },
            'quality_metrics': metrics
        }
    
        # Save analysis report
        report_path = Path('vocabs') / f'{pack_name}_analysis.json'
        with open(report_path, 'w') as f:
            json.dump(analysis, f, indent=2)
    
        logger.info(f"Analysis report saved to {report_path}")
    
        return analysis
    
    def create_pack(self, languages: List[str], pack_name: str, 
                   analyze_quality: bool = True) -> Dict[str, Any]:
        """
        Create optimized vocabulary pack for language group with quality analysis.
        
        This method performs comprehensive analysis of the corpus to create an
        optimized vocabulary pack with intelligent token selection and compression.
        
        Args:
            languages: List of language codes (e.g., ['en', 'es', 'fr'])
            pack_name: Name for the vocabulary pack
            
        Returns:
            Dictionary containing the vocabulary pack data with metadata
            
        Raises:
            ValueError: If invalid language codes provided
            RuntimeError: If vocabulary creation fails
        """
        logger.info(f"Creating vocabulary pack '{pack_name}' for languages: {languages}")
        
        try:
            # Validate languages
            self._validate_languages(languages)
            
            # 1. Analyze corpus to find common tokens
            logger.info("Analyzing corpus for token frequencies...")
            token_frequencies = self._analyze_corpus(languages)
            
            # 2. Select optimal vocabulary
            logger.info("Selecting optimal vocabulary...")
            vocab = self._select_vocabulary(token_frequencies)
            
            # 3. Create subword vocabulary for unknowns
            logger.info("Creating subword vocabulary...")
            subwords = self._create_subword_vocab(vocab, token_frequencies)
            
            # 4. Optimize token IDs for compression
            logger.info("Optimizing token IDs for compression...")
            optimized_vocab = self._optimize_token_ids(vocab, subwords)
            
            # 5. Calculate statistics
            logger.info("Calculating vocabulary statistics...")
            stats = self._calculate_stats(optimized_vocab, languages)
            
            # 6. Create pack structure
            version = self._get_pack_version(pack_name)  # Get dynamic version
            pack = {
                'name': pack_name,
                'version': version,
                'languages': languages,
                'tokens': optimized_vocab['tokens'],
                'subwords': optimized_vocab['subwords'],
                'special_tokens': optimized_vocab['special_tokens'],
                'metadata': {
                    'total_tokens': stats.total_tokens,
                    'coverage_percentage': stats.coverage_percentage,
                    'size_mb': stats.size_mb,
                    'compression_ratio': stats.compression_ratio,
                    'oov_rate': stats.oov_rate,
                    'config': self.config.__dict__
                }
            }
            
            # 7. Save pack
            self._save_pack(pack, pack_name)
            
            logger.info(f"Successfully created vocabulary pack '{pack_name}'")
            logger.info(f"Stats: {stats.total_tokens} tokens, {stats.coverage_percentage:.2f}% coverage, {stats.size_mb:.2f}MB")
            
            return pack

            # After creating the pack, analyze quality if requested
            if analyze_quality:
                try:
                    # Find a test corpus
                    test_corpus = None
                    for lang in languages:
                        if lang in self.corpus_paths:
                            test_corpus = self.corpus_paths[lang]
                            break
            
                    if test_corpus:
                        quality_analysis = self.analyze_vocabulary_quality(pack_name, test_corpus)
                
                        # Add quality metrics to pack metadata
                        pack['metadata']['quality_metrics'] = quality_analysis.get('quality_metrics', {})
                
                        # Re-save pack with quality metrics
                        self._save_pack(pack, pack_name)
                except Exception as e:
                    logger.warning(f"Quality analysis failed: {e}")
    
            return pack
            
        except Exception as e:
            logger.error(f"Failed to create vocabulary pack: {e}")
            raise RuntimeError(f"Vocabulary pack creation failed: {e}") from e
    
    def _validate_corpus_files(self) -> None:
        """Validate that all corpus files exist."""
        missing_files = []
        for lang, path in self.corpus_paths.items():
            if not Path(path).exists():
                missing_files.append(f"{lang}: {path}")
        
        if missing_files:
            raise FileNotFoundError(f"Missing corpus files: {missing_files}")
    
    def _validate_languages(self, languages: List[str]) -> None:
        """Validate that all requested languages have corpus files."""
        missing_langs = [lang for lang in languages if lang not in self.corpus_paths]
        if missing_langs:
            raise ValueError(f"No corpus files for languages: {missing_langs}")
    
    def _analyze_corpus(self, languages: List[str]) -> Counter:
        """
        Analyze corpus to find token frequencies across languages.
        
        Args:
            languages: List of language codes to analyze
            
        Returns:
            Counter with token frequencies
        """
        token_freq = Counter()
        total_lines = 0
        
        for lang in languages:
            corpus_path = self.corpus_paths[lang]
            lang_lines = 0
            
            try:
                with open(corpus_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:  # Skip empty lines
                            # Simple whitespace tokenization for frequency analysis
                            tokens = line.lower().split()
                            token_freq.update(tokens)
                            lang_lines += 1
                            
                logger.info(f"Processed {lang_lines:,} lines from {lang}")
                total_lines += lang_lines
                
            except IOError as e:
                logger.error(f"Error reading corpus for {lang}: {e}")
                raise
        
        logger.info(f"Total corpus analysis: {total_lines:,} lines, {len(token_freq):,} unique tokens")
        return token_freq
    
    def _select_vocabulary(self, token_frequencies: Counter) -> Dict[str, int]:
        """
        Select optimal vocabulary based on frequency analysis.
        
        Args:
            token_frequencies: Counter with token frequencies
            
        Returns:
            Dictionary mapping tokens to IDs
        """
        # Start with special tokens
        vocab = {
            '<pad>': 0,
            '<unk>': 1,
            '<s>': 2,
            '</s>': 3
        }
        
        # Filter tokens by minimum frequency
        filtered_tokens = {
            token: freq for token, freq in token_frequencies.items()
            if freq >= self.config.min_token_frequency
        }
        
        # Select top tokens by frequency
        available_slots = self.config.target_size - len(vocab)
        subword_slots = int(available_slots * self.config.subword_ratio)
        token_slots = available_slots - subword_slots
        
        # Get most frequent tokens
        top_tokens = token_frequencies.most_common(token_slots)
        
        # Add to vocabulary
        for token, freq in top_tokens:
            if len(vocab) < self.config.target_size - subword_slots:
                vocab[token] = len(vocab)
        
        logger.info(f"Selected {len(vocab)} primary tokens")
        return vocab
    
    def _create_subword_vocab(
        self, 
        vocab: Dict[str, int], 
        token_frequencies: Counter
    ) -> Dict[str, int]:
        """
        Create subword vocabulary for handling unknown tokens.
        
        Args:
            vocab: Primary vocabulary
            token_frequencies: Token frequency counter
            
        Returns:
            Dictionary mapping subword tokens to IDs
        """
        subwords = {}
        
        # Find common substrings in frequent tokens
        substring_freq = Counter()
        
        for token, freq in token_frequencies.most_common(10000):
            if token not in vocab:
                # Generate substrings
                for i in range(len(token)):
                    for j in range(i + 2, min(len(token) + 1, i + 8)):  # 2-7 char substrings
                        substring = token[i:j]
                        if len(substring) >= 2:
                            substring_freq[f"##{substring}"] += freq
        
        # Select top subwords
        available_slots = self.config.target_size - len(vocab)
        top_subwords = substring_freq.most_common(available_slots)
        
        for subword, freq in top_subwords:
            if len(subwords) < available_slots:
                subwords[subword] = len(vocab) + len(subwords)
        
        logger.info(f"Created {len(subwords)} subword tokens")
        return subwords
    
    def _optimize_token_ids(
        self, 
        vocab: Dict[str, int], 
        subwords: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Optimize token IDs for better compression.
        
        Args:
            vocab: Primary vocabulary
            subwords: Subword vocabulary
            
        Returns:
            Dictionary with optimized token mappings
        """
        # Combine vocabularies
        all_tokens = {**vocab, **subwords}
        
        # Sort by frequency (most frequent get lower IDs for better compression)
        # This is a simplified version - real implementation would use actual frequencies
        sorted_tokens = sorted(all_tokens.items(), key=lambda x: x[1])
        
        # Create optimized mapping
        optimized_tokens = {}
        optimized_subwords = {}
        special_tokens = {}
        
        for i, (token, old_id) in enumerate(sorted_tokens):
            if token.startswith('##'):
                optimized_subwords[token] = i
            elif token.startswith('<') and token.endswith('>'):
                special_tokens[token] = i
            else:
                optimized_tokens[token] = i
        
        return {
            'tokens': optimized_tokens,
            'subwords': optimized_subwords,
            'special_tokens': special_tokens
        }
    
    def _calculate_stats(
        self, 
        optimized_vocab: Dict[str, Any], 
        languages: List[str]
    ) -> VocabStats:
        """
        Calculate vocabulary statistics and coverage.
        
        Args:
            optimized_vocab: Optimized vocabulary mapping
            languages: List of target languages
            
        Returns:
            VocabStats object with coverage and performance metrics
        """
        # Calculate total tokens
        total_tokens = (
            len(optimized_vocab['tokens']) + 
            len(optimized_vocab['subwords']) + 
            len(optimized_vocab['special_tokens'])
        )
        
        # Calculate size
        packed_data = msgpack.packb(optimized_vocab)
        size_mb = len(packed_data) / (1024 * 1024)
        
        # Estimate coverage (simplified)
        coverage_percentage = min(95.0, (total_tokens / 30000) * 100)
        
        # Calculate compression ratio
        json_size = len(json.dumps(optimized_vocab).encode('utf-8'))
        compression_ratio = json_size / len(packed_data)
        
        # Estimate OOV rate
        oov_rate = max(0.01, (30000 - total_tokens) / 30000)
        
        return VocabStats(
            total_tokens=total_tokens,
            coverage_percentage=coverage_percentage,
            size_mb=size_mb,
            compression_ratio=compression_ratio,
            oov_rate=oov_rate
        )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _save_pack(self, pack: Dict[str, Any], pack_name: str) -> None:
        """
        Save vocabulary pack in multiple formats.
        
        Args:
            pack: Vocabulary pack data
            pack_name: Name for the pack files
        """
        # Create output directory
        output_dir = Path('vocabs')
        output_dir.mkdir(exist_ok=True)

        # Use version from pack data
        version = pack.get('version', '1.0')

        # Save JSON format with version
        json_path = output_dir / f'{pack_name}_v{version}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(pack, f, ensure_ascii=False, indent=2)
        
        # Save MessagePack format with version
        msgpack_path = output_dir / f'{pack_name}_v{version}.msgpack'
        with open(msgpack_path, 'wb') as f:
            f.write(msgpack.packb(pack))

        # Validate the saved pack
        is_valid, errors = self.validate_pack(str(json_path))
        if not is_valid:
            logger.warning(f"Pack validation warnings for {pack_name}: {errors}")

        logger.info(f"Saved vocabulary pack to {json_path} and {msgpack_path}")

    def validate_pack(self, pack_path: str) -> Tuple[bool, List[str]]:
        """
        Validate pack integrity and compatibility.
        
        Args:
            pack_path: Path to pack file (JSON or MessagePack)
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        pack_file = Path(pack_path)
        
        if not pack_file.exists():
            return False, ["Pack file does not exist"]
        
        try:
            # Load pack based on extension
            if pack_file.suffix == '.json':
                with open(pack_file, 'r', encoding='utf-8') as f:
                    pack = json.load(f)
            elif pack_file.suffix == '.msgpack':
                with open(pack_file, 'rb') as f:
                    pack = msgpack.unpackb(f.read(), strict_map_key=False)
            else:
                return False, ["Unsupported file format"]
            
            # Check required fields
            required_fields = ['name', 'version', 'languages', 'tokens', 'subwords', 'special_tokens', 'metadata']
            for field in required_fields:
                if field not in pack:
                    errors.append(f"Missing required field: {field}")
            
            # Validate structure
            if 'tokens' in pack and not isinstance(pack['tokens'], dict):
                errors.append("'tokens' must be a dictionary")
            
            if 'languages' in pack and not isinstance(pack['languages'], list):
                errors.append("'languages' must be a list")
            
            # Check token integrity
            if 'tokens' in pack:
                token_count = len(pack.get('tokens', {}))
                if token_count == 0:
                    errors.append("Pack contains no tokens")
                elif token_count < 1000:
                    errors.append(f"Suspiciously low token count: {token_count}")
            
            # Check special tokens
            if 'special_tokens' in pack:
                required_special = ['<pad>', '<unk>', '<s>', '</s>']
                for special in required_special:
                    if special not in pack['special_tokens']:
                        errors.append(f"Missing required special token: {special}")
            
            # Validate metadata
            if 'metadata' in pack:
                metadata = pack['metadata']
                
                # Check if metadata has expected fields
                expected_metadata = ['total_tokens', 'coverage_percentage', 'size_mb', 'compression_ratio', 'oov_rate']
                for field in expected_metadata:
                    if field not in metadata:
                        errors.append(f"Missing metadata field: {field}")
                
                # Validate token count consistency
                if 'total_tokens' in metadata:
                    reported_total = metadata['total_tokens']
                    actual_total = (
                        len(pack.get('tokens', {})) + 
                        len(pack.get('subwords', {})) + 
                        len(pack.get('special_tokens', {}))
                    )
                    if reported_total != actual_total:
                        errors.append(f"Token count mismatch: reported {reported_total}, actual {actual_total}")
                
                # Check ranges
                if 'coverage_percentage' in metadata:
                    coverage = metadata['coverage_percentage']
                    if not (0 <= coverage <= 100):
                        errors.append(f"Invalid coverage percentage: {coverage}")
                
                if 'oov_rate' in metadata:
                    oov_rate = metadata['oov_rate']
                    if not (0 <= oov_rate <= 1):
                        errors.append(f"Invalid OOV rate: {oov_rate}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            return False, [f"Error loading/parsing pack: {str(e)}"]

    def list_available_packs(self) -> List[str]:
        """
        List all available vocabulary packs.
        
        Returns:
            List of pack names
        """
        output_dir = Path('vocabs')
        if not output_dir.exists():
            return []
        
        packs = []
        for json_file in output_dir.glob('*_v*.json'):
            # Extract pack name without version
            pack_name = '_'.join(json_file.stem.split('_')[:-1])
            if pack_name and pack_name not in packs:
                packs.append(pack_name)
        return packs

    def _get_pack_version(self, pack_name: str) -> str:
        """
        Generate version based on existing packs.
        
        Args:
            pack_name: Name of the pack
            
        Returns:
            Version string (e.g., "1.0", "1.1", "2.0")
        """
        output_dir = Path('vocabs')
        if not output_dir.exists():
            return "1.0"
        
        existing_versions = []
        
        # Check for existing pack files
        for file in output_dir.glob(f'{pack_name}_v*.json'):
            # Extract version from filename (e.g., "latin_optimized_v1.2.json" -> "1.2")
            filename_parts = file.stem.split('_v')
            if len(filename_parts) >= 2:
                version_part = filename_parts[-1]
                try:
                    # Validate version format
                    parts = version_part.split('.')
                    if len(parts) == 2 and all(p.isdigit() for p in parts):
                        existing_versions.append(version_part)
                except ValueError:
                    continue
        
        if not existing_versions:
            return "1.0"
        
        # Find the latest version and increment
        latest = sorted(existing_versions, key=lambda v: [int(x) for x in v.split('.')])[-1]
        major, minor = map(int, latest.split('.'))
        
        # Increment minor version by default
        return f"{major}.{minor + 1}"

class VocabularyCreationProgress:
    """Track vocabulary creation progress"""
    
    def __init__(self):
        self.current_step = ""
        self.total_steps = 7
        self.current_step_num = 0
        self.progress_callbacks = []
    
    def update(self, step_num: int, step_name: str, details: str = ""):
        self.current_step_num = step_num
        self.current_step = step_name
        
        for callback in self.progress_callbacks:
            callback(step_num, self.total_steps, step_name, details)
    
    def add_callback(self, callback):
        self.progress_callbacks.append(callback)        


def main():
    """Example usage of the VocabularyPackCreator."""
    # Example corpus paths
    corpus_paths = {
        'en': 'data/en_corpus.txt',
        'es': 'data/es_corpus.txt',
        'fr': 'data/fr_corpus.txt',
        'de': 'data/de_corpus.txt'
    }
    
    # Custom configuration
    config = VocabConfig(
        target_size=25000,
        min_token_frequency=5,
        subword_ratio=0.3
    )
    
    try:
        # Create vocabulary pack creator
        creator = VocabularyPackCreator(corpus_paths, config)
        
        # Create Latin language pack
        pack = creator.create_pack(['en', 'es', 'fr', 'de'], 'latin_optimized')
        
        print(f"Created pack: {pack['name']}")
        print(f"Version: {pack['version']}")
        print(f"Languages: {pack['languages']}")
        print(f"Total tokens: {pack['metadata']['total_tokens']}")
        print(f"Coverage: {pack['metadata']['coverage_percentage']:.2f}%")
        print(f"Size: {pack['metadata']['size_mb']:.2f}MB")

        # Validate the created pack
        print("\nValidating pack...")
        pack_path = f"vocabs/{pack['name']}_v{pack['version']}.json"
        is_valid, errors = creator.validate_pack(pack_path)
        if is_valid:
            print("✅ Pack is valid")
        else:
            print(f"⚠️  Pack validation errors: {errors}")
        
        # List all available packs
        print("\nAvailable packs:")
        available_packs = creator.list_available_packs()
        for pack_name in available_packs:
            print(f"  - {pack_name}")
        
    except Exception as e:
        logger.error(f"Failed to create vocabulary pack: {e}")


if __name__ == "__main__":
    main()