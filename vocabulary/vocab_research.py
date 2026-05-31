import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.exceptions import VocabularyError

logger = logging.getLogger("vocabulary")

spm = None


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
