"""
Tests for vocabulary.vocab_config - vocabulary configuration data structures.
"""

import dataclasses
import pytest
from pipeline.vocabulary.config import CreationMode, UnifiedVocabConfig, VocabStats, LanguageGroup


class TestCreationMode:
    def test_enum_values(self):
        assert CreationMode.PRODUCTION.value == "production"
        assert CreationMode.RESEARCH.value == "research"
        assert CreationMode.HYBRID.value == "hybrid"

    def test_enum_members(self):
        members = {m.name for m in CreationMode}
        assert members == {"PRODUCTION", "RESEARCH", "HYBRID"}
        assert len(CreationMode) == 3


class TestUnifiedVocabConfig:
    def test_default_values(self):
        config = UnifiedVocabConfig()
        assert config.vocab_size == 25000
        assert config.model_type == "bpe"
        assert config.character_coverage == 0.9995
        assert config.num_threads == 16
        assert config.pad_id == 0
        assert config.unk_id == 1
        assert config.bos_id == 2
        assert config.eos_id == 3
        assert config.split_by_whitespace is True
        assert config.max_sentence_length == 4192
        assert config.shuffle_input_sentence is True
        assert config.min_token_frequency == 10
        assert config.subword_ratio == 0.3
        assert config.compression_level == 9
        assert config.analyze_quality is True
        assert config.quality_sample_size == 10000
        assert config.allow_evolution is True
        assert config.evolution_threshold == 0.1

    def test_custom_values(self):
        config = UnifiedVocabConfig(
            vocab_size=50000,
            model_type="unigram",
            character_coverage=0.9999,
            num_threads=8,
            min_token_frequency=20,
            analyze_quality=False,
            allow_evolution=False,
        )
        assert config.vocab_size == 50000
        assert config.model_type == "unigram"
        assert config.character_coverage == 0.9999
        assert config.num_threads == 8
        assert config.min_token_frequency == 20
        assert config.analyze_quality is False
        assert config.allow_evolution is False

    @pytest.mark.parametrize("field,value", [
        ("vocab_size", 1000),
        ("pad_id", 5),
        ("unk_id", 6),
        ("bos_id", 7),
        ("eos_id", 8),
        ("compression_level", 1),
        ("max_sentence_length", 256),
        ("quality_sample_size", 500),
    ])
    def test_individual_fields(self, field, value):
        config = UnifiedVocabConfig(**{field: value})
        assert getattr(config, field) == value

    def test_production_oriented(self):
        config = UnifiedVocabConfig(model_type="bpe", vocab_size=32000)
        assert config.model_type == "bpe"
        assert config.split_by_whitespace is True

    def test_research_oriented(self):
        config = UnifiedVocabConfig(
            model_type="unigram", vocab_size=64000,
            min_token_frequency=5, subword_ratio=0.5, compression_level=6,
        )
        assert config.min_token_frequency == 5
        assert config.subword_ratio == 0.5

    def test_hybrid_oriented(self):
        config = UnifiedVocabConfig(
            vocab_size=48000, min_token_frequency=15,
            analyze_quality=True, evolution_threshold=0.05,
        )
        assert config.evolution_threshold == 0.05


class TestVocabStats:
    def test_required_only(self):
        stats = VocabStats(
            total_tokens=100000, coverage_percentage=98.5,
            size_mb=45.2, compression_ratio=3.5, oov_rate=0.02,
        )
        assert stats.total_tokens == 100000
        assert stats.coverage_percentage == 98.5
        assert stats.size_mb == 45.2
        assert stats.compression_ratio == 3.5
        assert stats.oov_rate == 0.02
        assert stats.unigram_coverage is None
        assert stats.bigram_coverage is None
        assert stats.fertility is None
        assert stats.ambiguity is None

    def test_all_fields(self):
        stats = VocabStats(
            total_tokens=250000, coverage_percentage=99.2,
            size_mb=120.0, compression_ratio=4.1, oov_rate=0.01,
            unigram_coverage=0.95, bigram_coverage=0.85,
            fertility=1.2, ambiguity=0.3,
        )
        assert stats.unigram_coverage == 0.95
        assert stats.bigram_coverage == 0.85
        assert stats.fertility == 1.2
        assert stats.ambiguity == 0.3

    @pytest.mark.parametrize("field,value", [
        ("total_tokens", 0),
        ("coverage_percentage", 0.0),
        ("size_mb", 0.0),
        ("compression_ratio", 1.0),
        ("oov_rate", 1.0),
        ("unigram_coverage", 0.0),
        ("bigram_coverage", 1.0),
        ("fertility", 0.0),
        ("ambiguity", 0.0),
    ])
    def test_edge_values(self, field, value):
        kwargs = {
            "total_tokens": 1000, "coverage_percentage": 50.0,
            "size_mb": 10.0, "compression_ratio": 2.0, "oov_rate": 0.1,
            field: value,
        }
        stats = VocabStats(**kwargs)
        assert getattr(stats, field) == value


class TestLanguageGroup:
    def test_create(self):
        group = LanguageGroup(
            name="Romance",
            languages=["fr", "es", "it", "pt", "ro"],
            description="Romance languages",
        )
        assert group.name == "Romance"
        assert group.languages == ["fr", "es", "it", "pt", "ro"]
        assert group.description == "Romance languages"
        assert group.recommended_mode == CreationMode.PRODUCTION

    def test_custom_mode(self):
        group = LanguageGroup(
            name="Low Resource",
            languages=["zu", "xh"],
            description="Low-resource languages",
            recommended_mode=CreationMode.RESEARCH,
        )
        assert group.recommended_mode == CreationMode.RESEARCH

    def test_single_language(self):
        group = LanguageGroup(name="English", languages=["en"], description="English")
        assert group.languages == ["en"]
