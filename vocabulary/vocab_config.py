from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


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
