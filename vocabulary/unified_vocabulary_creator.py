# vocabulary/unified_vocabulary_creator.py
"""
Unified vocabulary pack creator supporting both production (SentencePiece) 
and research (frequency analysis) approaches.

Combines and replaces:
- create_vocabulary_packs.py
- create_vocabulary_packs_from_data.py
"""

import json
import logging
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum

import msgpack
import numpy as np
# SentencePiece is optional; import lazily only when needed (production mode)
spm = None  # type: ignore
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from utils.exceptions import DataError, VocabularyError

# Configure centralized logging for vocabulary
from utils.logging_config import setup_logging
setup_logging(log_dir="logs", log_level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("vocabulary")


class CreationMode(Enum):
    """Vocabulary creation modes"""
    PRODUCTION = "production"      # SentencePiece-based (Google standard)
    RESEARCH = "research"          # Frequency analysis-based
    HYBRID = "hybrid"             # Combine both approaches


@dataclass
class UnifiedVocabConfig:
    """Unified configuration for all creation modes"""
    # Common settings
    vocab_size: int = 25000
    model_type: str = 'bpe'
    character_coverage: float = 0.9995
    num_threads: int = 16
    
    # SentencePiece settings (PRODUCTION mode)
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3
    split_by_whitespace: bool = True
    max_sentence_length: int = 4192
    shuffle_input_sentence: bool = True
    
    # Frequency analysis settings (RESEARCH mode)
    min_token_frequency: int = 10
    subword_ratio: float = 0.3
    compression_level: int = 9
    
    # Quality analysis settings
    analyze_quality: bool = True
    quality_sample_size: int = 10000
    
    # Evolution settings
    allow_evolution: bool = True
    evolution_threshold: float = 0.1  # Min improvement needed


@dataclass
class VocabStats:
    """Unified statistics for vocabulary packs"""
    total_tokens: int
    coverage_percentage: float
    size_mb: float
    compression_ratio: float
    oov_rate: float
    
    # Quality metrics (optional)
    unigram_coverage: Optional[float] = None
    bigram_coverage: Optional[float] = None
    fertility: Optional[float] = None
    ambiguity: Optional[float] = None


@dataclass
class LanguageGroup:
    """Language group definition"""
    name: str
    languages: List[str]
    description: str
    recommended_mode: CreationMode = CreationMode.PRODUCTION


class UnifiedVocabularyCreator:
    """
    Unified vocabulary pack creator supporting multiple creation strategies.
    
    This class combines production (SentencePiece) and research (frequency analysis)
    approaches, allowing users to choose the best method for their use case.
    """
    
    def __init__(
        self,
        corpus_dir: str = 'data/processed',
        output_dir: str = 'vocabs',
        config: Optional[UnifiedVocabConfig] = None,
        default_mode: CreationMode = CreationMode.PRODUCTION
    ):
        """
        Initialize unified vocabulary creator.
        
        Args:
            corpus_dir: Directory containing corpus files
            output_dir: Directory for output vocabulary packs
            config: Unified configuration
            default_mode: Default creation mode
        """
        self.corpus_dir = Path(corpus_dir)
        self.output_dir = Path(output_dir)
        self.config = config or UnifiedVocabConfig()
        self.default_mode = default_mode
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.tokenizer = None  # Lazy init for SentencePiece
        self.language_groups = self._define_language_groups()
        
        # Cache for corpus paths
        self.corpus_paths = {}
        self._discover_corpus_files()
        
        logger.info(f"UnifiedVocabularyCreator initialized")
        logger.info(f"Default mode: {default_mode.value}")
        logger.info(f"Corpus directory: {self.corpus_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _discover_corpus_files(self):
        """Discover available corpus files"""
        for corpus_file in self.corpus_dir.glob('*_corpus.txt'):
            # Extract language code from filename
            lang = corpus_file.stem.replace('_corpus', '')
            if '_' not in lang:  # Skip domain-specific files for now
                self.corpus_paths[lang] = str(corpus_file)
        
        logger.info(f"Discovered corpus files for {len(self.corpus_paths)} languages")
    
    def _define_language_groups(self) -> Dict[str, LanguageGroup]:
        """Define language groups with recommended modes"""
        return {
            'latin': LanguageGroup(
                name='latin',
                languages=['en', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'sv', 'pl'],
                description='Latin script languages',
                recommended_mode=CreationMode.PRODUCTION
            ),
            'cjk': LanguageGroup(
                name='cjk',
                languages=['zh', 'ja', 'ko'],
                description='Chinese, Japanese, Korean',
                recommended_mode=CreationMode.PRODUCTION  # SentencePiece handles CJK well
            ),
            'arabic': LanguageGroup(
                name='arabic',
                languages=['ar'],
                description='Arabic script languages',
                recommended_mode=CreationMode.PRODUCTION
            ),
            'devanagari': LanguageGroup(
                name='devanagari',
                languages=['hi'],
                description='Devanagari script languages',
                recommended_mode=CreationMode.PRODUCTION
            ),
            'cyrillic': LanguageGroup(
                name='cyrillic',
                languages=['ru', 'uk'],
                description='Cyrillic script languages',
                recommended_mode=CreationMode.PRODUCTION
            ),
            'research': LanguageGroup(
                name='research',
                languages=['en'],  # Can be customized
                description='Research and experimentation',
                recommended_mode=CreationMode.RESEARCH
            )
        }
    
    def create_all_packs(
        self,
        mode: Optional[CreationMode] = None,
        groups_to_create: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Create all or specified vocabulary packs.
        
        Args:
            mode: Override creation mode (None = use recommended)
            groups_to_create: List of group names (None = all)
        
        Returns:
            Dictionary of created packs
        """
        created_packs = {}
        failed_packs = []
        
        groups = groups_to_create or list(self.language_groups.keys())
        
        for group_name in groups:
            if group_name not in self.language_groups:
                logger.warning(f"Unknown group: {group_name}")
                continue
            
            group = self.language_groups[group_name]
            
            # Use recommended mode unless overridden
            creation_mode = mode or group.recommended_mode
            
            try:
                logger.info(f"\n📦 Creating {group_name} pack using {creation_mode.value} mode...")
                pack = self.create_pack(
                    pack_name=group_name,
                    languages=group.languages,
                    mode=creation_mode
                )
                created_packs[group_name] = pack
                
            except Exception as e:
                logger.error(f"Failed to create {group_name}: {e}")
                failed_packs.append(group_name)
        
        # Summary
        logger.info(f"\n📊 Summary:")
        logger.info(f"Successfully created: {len(created_packs)} packs")
        if failed_packs:
            logger.warning(f"Failed: {failed_packs}")
        
        return created_packs
    
    def create_pack(
        self,
        pack_name: str,
        languages: List[str],
        mode: Optional[CreationMode] = None,
        domain: Optional[str] = None,
        base_pack_path: Optional[str] = None,
        tokens_to_add: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create or evolve a vocabulary pack using specified mode.
        
        This is the main unified method that routes to appropriate creation strategy.
        
        Args:
            pack_name: Name for the vocabulary pack
            languages: List of language codes
            mode: Creation mode (None = use default)
            domain: Optional domain specification
            base_pack_path: Path for evolution mode
            tokens_to_add: Additional tokens for evolution
        
        Returns:
            Created vocabulary pack
        """
        mode = mode or self.default_mode
        
        logger.info(f"Creating pack '{pack_name}' using {mode.value} mode")
        logger.info(f"Languages: {languages}")
        if domain:
            logger.info(f"Domain: {domain}")
        
        try:
            # Evolution mode
            if base_pack_path and self.config.allow_evolution:
                return self._evolve_pack(
                    pack_name, base_pack_path, tokens_to_add, mode
                )
            
            # Route to appropriate creation method
            if mode == CreationMode.PRODUCTION:
                vocab_data = self._create_sentencepiece_vocab(
                    languages, pack_name, domain
                )
            elif mode == CreationMode.RESEARCH:
                vocab_data = self._create_frequency_vocab(
                    languages, pack_name, domain
                )
            elif mode == CreationMode.HYBRID:
                vocab_data = self._create_hybrid_vocab(
                    languages, pack_name, domain
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            # Create pack structure
            pack = self._create_pack_structure(
                pack_name, languages, vocab_data, mode, domain
            )
            
            # Quality analysis if enabled
            if self.config.analyze_quality:
                quality_metrics = self._analyze_quality(pack, languages)
                pack['metadata']['quality_metrics'] = quality_metrics
            
            # Save pack
            self._save_pack(pack, pack_name)
            
            logger.info(f"✅ Successfully created pack '{pack_name}'")
            self._log_pack_stats(pack)
            
            return pack
            
        except Exception as e:
            logger.error(f"Failed to create pack {pack_name}: {e}")
            raise VocabularyError(f"Pack creation failed: {e}") from e
    
    # ============= PRODUCTION MODE (SentencePiece) =============
    
    def _create_sentencepiece_vocab(
        self,
        languages: List[str],
        pack_name: str,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create vocabulary using SentencePiece (production mode)"""
        logger.info("Using SentencePiece for vocabulary creation...")
        
        # Merge corpora
        merged_corpus = self._merge_corpora(languages, pack_name, domain)
        
        # Train SentencePiece model
        model_path = self._train_sentencepiece_model(merged_corpus, pack_name)
        
        # Create vocabulary mappings
        vocab_data = self._create_vocabulary_mappings(model_path, languages)
        
        # Extract embeddings if available
        if hasattr(self, 'extract_embeddings_for_tokens'):
            embeddings = self.extract_embeddings_for_tokens(vocab_data['tokens'])
            vocab_data['embeddings'] = embeddings
        
        # Cleanup
        self._cleanup_temp_files(merged_corpus, model_path)
        
        return vocab_data
    
    def _train_sentencepiece_model(self, corpus_file: str, pack_name: str) -> str:
        """Train SentencePiece model"""
        model_prefix = str(self.output_dir / f"temp_{pack_name}")
        
        train_args = [
            f'--input={corpus_file}',
            f'--model_prefix={model_prefix}',
            f'--vocab_size={self.config.vocab_size}',
            f'--model_type={self.config.model_type}',
            f'--character_coverage={self.config.character_coverage}',
            f'--pad_id={self.config.pad_id}',
            f'--unk_id={self.config.unk_id}',
            f'--bos_id={self.config.bos_id}',
            f'--eos_id={self.config.eos_id}',
            f'--num_threads={self.config.num_threads}',
            f'--train_extremely_large_corpus=true',
            f'--byte_fallback=true'
        ]
        
        # Lazy import to avoid requiring sentencepiece when not used
        global spm
        if spm is None:
            try:
                import sentencepiece as spm  # type: ignore
            except Exception as e:
                raise VocabularyError("sentencepiece not installed; required for PRODUCTION mode. Install with: pip install sentencepiece") from e

        logger.info("Training SentencePiece model...")
        spm.SentencePieceTrainer.train(' '.join(train_args))
        
        model_path = f"{model_prefix}.model"
        if not Path(model_path).exists():
            raise VocabularyError("Model file not created")
        
        return model_path
    
    def _create_vocabulary_mappings(
        self, 
        model_path: str, 
        languages: List[str]
    ) -> Dict[str, Any]:
        """Create vocabulary mappings from SentencePiece model"""
        # Lazy import to avoid requiring sentencepiece when not used
        global spm
        if spm is None:
            try:
                import sentencepiece as spm  # type: ignore
            except Exception as e:
                raise VocabularyError("sentencepiece not installed; required for PRODUCTION mode. Install with: pip install sentencepiece") from e
        
        sp = spm.SentencePieceProcessor()
        sp.load(model_path)
        
        tokens = {}
        subwords = {}
        special_tokens = {}
        
        for i in range(sp.get_piece_size()):
            piece = sp.id_to_piece(i)
            
            if piece in ['<pad>', '<unk>', '<s>', '</s>']:
                special_tokens[piece] = i
            elif piece.startswith('▁'):
                clean_piece = piece[1:] if len(piece) > 1 else piece
                tokens[clean_piece] = i
            elif piece.startswith('##'):
                subwords[piece] = i
            else:
                if len(piece) > 1 and not piece.isalnum():
                    subwords[piece] = i
                else:
                    tokens[piece] = i
        
        # Add language tokens
        next_id = sp.get_piece_size()
        for lang in languages:
            lang_token = f'<{lang}>'
            if lang_token not in special_tokens:
                special_tokens[lang_token] = next_id
                next_id += 1
        
        logger.info(f"Created {len(tokens)} tokens, {len(subwords)} subwords")
        
        return {
            'tokens': tokens,
            'subwords': subwords,
            'special_tokens': special_tokens
        }
    
    # ============= RESEARCH MODE (Frequency Analysis) =============
    
    def _create_frequency_vocab(
        self,
        languages: List[str],
        pack_name: str,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create vocabulary using frequency analysis (research mode)"""
        logger.info("Using frequency analysis for vocabulary creation...")
        
        # Update corpus paths for domain if needed
        if domain:
            self._update_corpus_paths_for_domain(languages, domain)
        
        # Analyze corpus
        token_frequencies = self._analyze_corpus(languages)
        
        # Select vocabulary
        vocab = self._select_vocabulary(token_frequencies)
        
        # Create subwords
        subwords = self._create_subword_vocab(vocab, token_frequencies)
        
        # Optimize token IDs
        optimized = self._optimize_token_ids(vocab, subwords)
        
        return optimized
    
    def _analyze_corpus(self, languages: List[str]) -> Counter:
        """Analyze corpus for token frequencies"""
        token_freq = Counter()
        
        for lang in languages:
            if lang not in self.corpus_paths:
                logger.warning(f"No corpus for {lang}")
                continue
            
            corpus_path = self.corpus_paths[lang]
            
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    tokens = line.lower().split()
                    token_freq.update(tokens)
        
        logger.info(f"Analyzed {len(token_freq)} unique tokens")
        return token_freq
    
    def _select_vocabulary(self, token_frequencies: Counter) -> Dict[str, int]:
        """Select optimal vocabulary based on frequencies"""
        vocab = {
            '<pad>': 0,
            '<unk>': 1,
            '<s>': 2,
            '</s>': 3
        }
        
        # Filter by minimum frequency
        filtered = {
            token: freq for token, freq in token_frequencies.items()
            if freq >= self.config.min_token_frequency
        }
        
        # Calculate slots
        available_slots = self.config.vocab_size - len(vocab)
        subword_slots = int(available_slots * self.config.subword_ratio)
        token_slots = available_slots - subword_slots
        
        # Select top tokens
        for token, _ in token_frequencies.most_common(token_slots):
            if len(vocab) < self.config.vocab_size - subword_slots:
                vocab[token] = len(vocab)
        
        return vocab
    
    def _create_subword_vocab(
        self,
        vocab: Dict[str, int],
        token_frequencies: Counter
    ) -> Dict[str, int]:
        """Create subword vocabulary"""
        subwords = {}
        substring_freq = Counter()
        
        # Find common substrings
        for token, freq in token_frequencies.most_common(10000):
            if token not in vocab:
                for i in range(len(token)):
                    for j in range(i + 2, min(len(token) + 1, i + 8)):
                        substring = token[i:j]
                        if len(substring) >= 2:
                            substring_freq[f"##{substring}"] += freq
        
        # Select top subwords
        available_slots = self.config.vocab_size - len(vocab)
        for subword, _ in substring_freq.most_common(available_slots):
            if len(subwords) < available_slots:
                subwords[subword] = len(vocab) + len(subwords)
        
        return subwords
    
    def _optimize_token_ids(
        self,
        vocab: Dict[str, int],
        subwords: Dict[str, int]
    ) -> Dict[str, Any]:
        """Optimize token IDs for compression"""
        all_tokens = {**vocab, **subwords}
        sorted_tokens = sorted(all_tokens.items(), key=lambda x: x[1])
        
        optimized_tokens = {}
        optimized_subwords = {}
        special_tokens = {}
        
        for i, (token, _) in enumerate(sorted_tokens):
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
    
    # ============= HYBRID MODE =============
    
    def _create_hybrid_vocab(
        self,
        languages: List[str],
        pack_name: str,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create vocabulary using both methods and merge"""
        logger.info("Using hybrid approach...")
        
        # Get SentencePiece vocabulary
        sp_vocab = self._create_sentencepiece_vocab(
            languages, f"{pack_name}_sp", domain
        )
        
        # Get frequency-based vocabulary
        freq_vocab = self._create_frequency_vocab(
            languages, f"{pack_name}_freq", domain
        )
        
        # Merge intelligently
        merged = self._merge_vocabularies(sp_vocab, freq_vocab)
        
        return merged
    
    def _merge_vocabularies(
        self,
        sp_vocab: Dict[str, Any],
        freq_vocab: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Intelligently merge two vocabularies"""
        # Take union of tokens, intersection of subwords
        merged_tokens = {**sp_vocab['tokens'], **freq_vocab['tokens']}
        
        # Keep subwords that appear in both
        merged_subwords = {}
        for subword in sp_vocab['subwords']:
            if subword in freq_vocab['subwords']:
                merged_subwords[subword] = len(merged_tokens) + len(merged_subwords)
        
        # Combine special tokens
        merged_special = {**sp_vocab['special_tokens'], **freq_vocab['special_tokens']}
        
        # Limit to target size
        if len(merged_tokens) > self.config.vocab_size * 0.7:
            # Keep most important tokens
            merged_tokens = dict(list(merged_tokens.items())[:int(self.config.vocab_size * 0.7)])
        
        return {
            'tokens': merged_tokens,
            'subwords': merged_subwords,
            'special_tokens': merged_special
        }
    
    # ============= EVOLUTION MODE =============
    
    def _evolve_pack(
        self,
        pack_name: str,
        base_pack_path: str,
        tokens_to_add: Optional[List[str]],
        mode: CreationMode
    ) -> Dict[str, Any]:
        """Evolve existing pack with new tokens"""
        logger.info(f"Evolving pack from {base_pack_path}")
        
        # Load base pack
        with open(base_pack_path, 'rb') as f:
            base_pack = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
        
        vocab_data = {
            'tokens': base_pack.get('tokens', {}),
            'subwords': base_pack.get('subwords', {}),
            'special_tokens': base_pack.get('special_tokens', {})
        }
        
        # Add new tokens
        if tokens_to_add:
            max_id = max([
                max(vocab_data['tokens'].values(), default=0),
                max(vocab_data['subwords'].values(), default=0),
                max(vocab_data['special_tokens'].values(), default=0)
            ])
            
            added = 0
            for token in tokens_to_add:
                if token not in vocab_data['tokens']:
                    max_id += 1
                    vocab_data['tokens'][token] = max_id
                    added += 1
            
            logger.info(f"Added {added} new tokens")
        
        # Re-analyze if needed
        if mode == CreationMode.RESEARCH and hasattr(self, '_analyze_corpus'):
            # Could re-optimize based on new corpus
            pass
        
        return vocab_data
    
    # ============= COMMON METHODS =============
    
    def _merge_corpora(
        self,
        languages: List[str],
        pack_name: str,
        domain: Optional[str] = None
    ) -> str:
        """Merge corpora from multiple languages"""
        merged_path = self.output_dir / f"temp_{pack_name}_corpus.txt"
        
        with open(merged_path, 'w', encoding='utf-8') as out_file:
            for lang in languages:
                # Determine corpus file
                if domain:
                    corpus_file = self.corpus_dir / f"{lang}_{domain}_corpus.txt"
                else:
                    corpus_file = Path(self.corpus_paths.get(
                        lang, 
                        self.corpus_dir / f"{lang}_corpus.txt"
                    ))
                
                if not corpus_file.exists():
                    logger.warning(f"Missing corpus for {lang}")
                    continue
                
                with open(corpus_file, 'r', encoding='utf-8') as in_file:
                    for line in in_file:
                        if line.strip():
                            out_file.write(line)
                
                logger.info(f"Merged corpus from {lang}")
        
        return str(merged_path)
    
    def _create_pack_structure(
        self,
        pack_name: str,
        languages: List[str],
        vocab_data: Dict[str, Any],
        mode: CreationMode,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create complete pack structure"""
        # Calculate stats
        stats = self._calculate_stats(vocab_data, languages)
        
        # Get version
        version = self._get_pack_version(pack_name)
        
        pack = {
            'name': pack_name,
            'version': version,
            'languages': languages,
            'tokens': vocab_data['tokens'],
            'subwords': vocab_data['subwords'],
            'special_tokens': vocab_data['special_tokens'],
            'metadata': {
                'total_tokens': stats.total_tokens,
                'coverage_percentage': stats.coverage_percentage,
                'size_mb': stats.size_mb,
                'compression_ratio': stats.compression_ratio,
                'oov_rate': stats.oov_rate,
                'creation_mode': mode.value,
                'domain': domain,
                'config': {
                    'vocab_size': self.config.vocab_size,
                    'model_type': self.config.model_type,
                    'character_coverage': self.config.character_coverage
                }
            }
        }
        
        # Add embeddings if available
        if 'embeddings' in vocab_data:
            pack['embeddings'] = vocab_data['embeddings']
            pack['metadata']['has_embeddings'] = True
        
        return pack
    
    def _calculate_stats(
        self,
        vocab_data: Dict[str, Any],
        languages: List[str]
    ) -> VocabStats:
        """Calculate vocabulary statistics"""
        total_tokens = sum(
            len(vocab_data.get(key, {}))
            for key in ['tokens', 'subwords', 'special_tokens']
        )
        
        # Pack and measure size
        packed = msgpack.packb(vocab_data)
        size_mb = len(packed) / (1024 * 1024)
        
        # Compression ratio
        json_size = len(json.dumps(vocab_data).encode())
        compression_ratio = json_size / len(packed) if packed else 1.0
        
        # Coverage estimate
        coverage = min(95.0, (total_tokens / self.config.vocab_size) * 100)
        
        # OOV estimate
        oov_rate = max(0.01, 1.0 - (total_tokens / self.config.vocab_size))
        
        return VocabStats(
            total_tokens=total_tokens,
            coverage_percentage=coverage,
            size_mb=size_mb,
            compression_ratio=compression_ratio,
            oov_rate=oov_rate
        )
    
    def _analyze_quality(
        self,
        pack: Dict[str, Any],
        languages: List[str]
    ) -> Dict[str, float]:
        """Analyze vocabulary quality"""
        metrics = {
            'unigram_coverage': 0.0,
            'bigram_coverage': 0.0,
            'fertility': 0.0,
            'ambiguity': 0.0
        }
        
        # Sample analysis on first available corpus
        for lang in languages:
            if lang in self.corpus_paths:
                corpus_path = self.corpus_paths[lang]
                # Simplified quality analysis
                all_vocab = {
                    **pack['tokens'],
                    **pack['subwords'],
                    **pack['special_tokens']
                }
                
                # Count coverage
                total_words = 0
                covered_words = 0
                
                with open(corpus_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= self.config.quality_sample_size:
                            break
                        
                        words = line.strip().split()
                        total_words += len(words)
                        
                        for word in words:
                            if word.lower() in all_vocab:
                                covered_words += 1
                
                if total_words > 0:
                    metrics['unigram_coverage'] = covered_words / total_words
                    metrics['bigram_coverage'] = metrics['unigram_coverage'] ** 2
                
                break
        
        return metrics
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(IOError)
    )
    def _save_pack(self, pack: Dict[str, Any], pack_name: str):
        """Save vocabulary pack with retry logic"""
        version = pack['version']
        
        # Save JSON
        json_path = self.output_dir / f'{pack_name}_v{version}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(pack, f, ensure_ascii=False, indent=2)
        
        # Save MessagePack
        msgpack_path = self.output_dir / f'{pack_name}_v{version}.msgpack'
        with open(msgpack_path, 'wb') as f:
            f.write(msgpack.packb(pack))
        
        logger.info(f"Saved pack to {json_path} and {msgpack_path}")
    
    def _cleanup_temp_files(self, *files):
        """Clean up temporary files"""
        for file_path in files:
            try:
                if Path(file_path).exists():
                    os.remove(file_path)
                    logger.debug(f"Cleaned up {file_path}")
            except OSError as e:
                logger.warning(f"Could not clean up {file_path}: {e}")
    
    def _get_pack_version(self, pack_name: str) -> str:
        """Generate version based on existing packs"""
        existing_versions = []
        
        for file in self.output_dir.glob(f'{pack_name}_v*.json'):
            version_part = file.stem.split('_v')[-1]
            try:
                existing_versions.append(version_part)
            except ValueError:
                continue
        
        if not existing_versions:
            return "1.0"
        
        latest = sorted(
            existing_versions,
            key=lambda v: [int(x) for x in v.split('.')]
        )[-1]
        major, minor = map(int, latest.split('.'))
        
        return f"{major}.{minor + 1}"
    
    def _update_corpus_paths_for_domain(
        self,
        languages: List[str],
        domain: str
    ):
        """Update corpus paths for domain-specific creation"""
        for lang in languages:
            domain_corpus = self.corpus_dir / f"{lang}_{domain}_corpus.txt"
            if domain_corpus.exists():
                self.corpus_paths[lang] = str(domain_corpus)
                logger.info(f"Using domain corpus for {lang}: {domain_corpus}")
    
    def _log_pack_stats(self, pack: Dict[str, Any]):
        """Log pack statistics"""
        metadata = pack['metadata']
        logger.info(f"Pack Statistics:")
        logger.info(f"  Total tokens: {metadata['total_tokens']:,}")
        logger.info(f"  Coverage: {metadata['coverage_percentage']:.2f}%")
        logger.info(f"  Size: {metadata['size_mb']:.2f}MB")
        logger.info(f"  OOV rate: {metadata['oov_rate']:.4f}")
        logger.info(f"  Creation mode: {metadata['creation_mode']}")
    
    # ============= UTILITY METHODS =============
    
    def validate_pack(self, pack_path: str) -> Tuple[bool, List[str]]:
        """Validate vocabulary pack integrity"""
        errors = []
        pack_file = Path(pack_path)
        
        if not pack_file.exists():
            return False, ["Pack file does not exist"]
        
        try:
            # Load pack
            if pack_file.suffix == '.json':
                with open(pack_file, 'r') as f:
                    pack = json.load(f)
            elif pack_file.suffix == '.msgpack':
                with open(pack_file, 'rb') as f:
                    pack = msgpack.unpackb(f.read(), strict_map_key=False)
            else:
                return False, ["Unsupported format"]
            
            # Check required fields
            required = [
                'name', 'version', 'languages', 
                'tokens', 'subwords', 'special_tokens', 'metadata'
            ]
            for field in required:
                if field not in pack:
                    errors.append(f"Missing field: {field}")
            
            # Validate special tokens
            if 'special_tokens' in pack:
                for token in ['<pad>', '<unk>', '<s>', '</s>']:
                    if token not in pack['special_tokens']:
                        errors.append(f"Missing special token: {token}")
            
            # Check token count
            if 'metadata' in pack and 'total_tokens' in pack['metadata']:
                reported = pack['metadata']['total_tokens']
                actual = sum(
                    len(pack.get(key, {}))
                    for key in ['tokens', 'subwords', 'special_tokens']
                )
                if reported != actual:
                    errors.append(f"Token count mismatch: {reported} vs {actual}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            return False, [f"Error loading pack: {e}"]
    
    def list_available_packs(self) -> List[Dict[str, Any]]:
        """List all available vocabulary packs with details"""
        packs = []
        
        for json_file in self.output_dir.glob('*_v*.json'):
            try:
                # Parse pack info
                parts = json_file.stem.split('_v')
                if len(parts) >= 2:
                    pack_name = '_'.join(parts[:-1])
                    version = parts[-1]
                    
                    # Load metadata
                    with open(json_file, 'r') as f:
                        pack_data = json.load(f)
                    
                    packs.append({
                        'name': pack_name,
                        'version': version,
                        'languages': pack_data.get('languages', []),
                        'creation_mode': pack_data.get('metadata', {}).get('creation_mode', 'unknown'),
                        'total_tokens': pack_data.get('metadata', {}).get('total_tokens', 0),
                        'path': str(json_file)
                    })
            except Exception as e:
                logger.warning(f"Could not read {json_file}: {e}")
        
        return sorted(packs, key=lambda x: (x['name'], x['version']))
    
    def compare_packs(
        self,
        pack1_path: str,
        pack2_path: str
    ) -> Dict[str, Any]:
        """Compare two vocabulary packs"""
        # Load packs
        with open(pack1_path, 'r') as f:
            pack1 = json.load(f)
        with open(pack2_path, 'r') as f:
            pack2 = json.load(f)
        
        # Compare
        comparison = {
            'pack1_name': pack1['name'],
            'pack2_name': pack2['name'],
            'token_overlap': len(
                set(pack1['tokens'].keys()) & set(pack2['tokens'].keys())
            ),
            'subword_overlap': len(
                set(pack1['subwords'].keys()) & set(pack2['subwords'].keys())
            ),
            'pack1_unique_tokens': len(
                set(pack1['tokens'].keys()) - set(pack2['tokens'].keys())
            ),
            'pack2_unique_tokens': len(
                set(pack2['tokens'].keys()) - set(pack1['tokens'].keys())
            ),
            'size_difference_mb': abs(
                pack1['metadata']['size_mb'] - pack2['metadata']['size_mb']
            )
        }
        
        return comparison

    def extract_embeddings_for_tokens(
        self,
        tokens: Dict[str, int]
    ) -> Optional[Dict[str, List[float]]]:
        """
        Extract or generate embeddings for tokens.
        Placeholder for production embedding extraction.
        """
        # This would integrate with your trained models
        # For now, returns None to indicate no embeddings
        logger.info("Embedding extraction not configured")
        return None