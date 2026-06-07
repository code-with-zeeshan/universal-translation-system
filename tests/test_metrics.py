"""
Tests for evaluation.metrics - translation data structures.
"""

import pytest
from evaluation.evaluator import TranslationPair


class TestTranslationPair:
    def test_create_source_target(self):
        pair = TranslationPair(source="Hello", target="Bonjour", source_lang="en", target_lang="fr")
        assert pair.source == "Hello"
        assert pair.target == "Bonjour"
        assert pair.source_lang == "en"
        assert pair.target_lang == "fr"
        assert pair.predicted is None

    def test_create_with_predicted(self):
        pair = TranslationPair(
            source="Good morning", target="Bonjour",
            source_lang="en", target_lang="fr", predicted="Bon matin",
        )
        assert pair.source == "Good morning"
        assert pair.target == "Bonjour"
        assert pair.source_lang == "en"
        assert pair.target_lang == "fr"
        assert pair.predicted == "Bon matin"

    def test_predicted_default_none(self):
        pair = TranslationPair(source="a", target="b", source_lang="en", target_lang="fr")
        assert pair.predicted is None

    def test_fields_are_strings(self):
        pair = TranslationPair(source="Hello", target="Hola", source_lang="en", target_lang="es")
        assert isinstance(pair.source, str)
        assert isinstance(pair.target, str)
        assert isinstance(pair.source_lang, str)
        assert isinstance(pair.target_lang, str)

    def test_update_predicted(self):
        pair = TranslationPair(source="Hello", target="Bonjour", source_lang="en", target_lang="fr")
        pair.predicted = "Bonjour"
        assert pair.predicted == "Bonjour"

    def test_equality(self):
        a = TranslationPair(source="Hello", target="Bonjour", source_lang="en", target_lang="fr")
        b = TranslationPair(source="Hello", target="Bonjour", source_lang="en", target_lang="fr")
        assert a == b

    def test_inequality(self):
        a = TranslationPair(source="Hello", target="Bonjour", source_lang="en", target_lang="fr")
        b = TranslationPair(source="Hello", target="Hola", source_lang="en", target_lang="es")
        assert a != b

    def test_empty_strings(self):
        pair = TranslationPair(source="", target="", source_lang="", target_lang="")
        assert pair.source == ""
        assert pair.target == ""

    def test_long_text(self):
        src = "The quick brown fox jumps " * 20
        tgt = "Le renard brun rapide saute " * 20
        pair = TranslationPair(source=src, target=tgt, source_lang="en", target_lang="fr")
        assert pair.source == src
        assert pair.target == tgt

    def test_repr(self):
        pair = TranslationPair(source="Hi", target="Salut", source_lang="en", target_lang="fr")
        r = repr(pair)
        assert "TranslationPair" in r
        assert "Hi" in r
        assert "Salut" in r

    def test_predicted_aliasing(self):
        a = TranslationPair(source="Hello", target="Bonjour", source_lang="en", target_lang="fr")
        b = TranslationPair(source="World", target="Monde", source_lang="en", target_lang="fr")
        assert a.predicted is None
        assert b.predicted is None
