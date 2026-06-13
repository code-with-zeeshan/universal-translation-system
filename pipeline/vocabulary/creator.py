# vocabulary/vocabulary_creator.py
"""
Unified vocabulary pack creator supporting both production (SentencePiece) 
and research (frequency analysis) approaches.

Combines and replaces:
- create_vocabulary_packs.py
- create_vocabulary_packs_from_data.py
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import msgpack

from utils.common_utils import RuntimeDirectoryManager

from utils.exceptions import VocabularyError
from utils.logging_config import setup_logging

from pipeline.vocabulary.config import UnifiedVocabConfig, VocabStats, LanguageGroup, CreationMode
from pipeline.vocabulary.production import _create_sentencepiece_vocab, _train_sentencepiece_model
from pipeline.vocabulary.research import (
    _create_frequency_vocab,
    _analyze_corpus,
    _select_vocabulary,
    _create_subword_vocab,
    _optimize_token_ids,
    _merge_vocabularies,
    _merge_corpora,
    _create_vocabulary_mappings,
)
from pipeline.vocabulary.validation import validate_pack, compare_packs, _get_pack_version, _save_pack, _cleanup_temp_files


setup_logging(log_dir=str(RuntimeDirectoryManager().logs_dir), log_level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("vocabulary")


class UnifiedVocabularyCreator:
    """
    Unified vocabulary pack creator supporting multiple creation strategies.
    
    This class combines production (SentencePiece) and research (frequency analysis)
    approaches, allowing users to choose the best method for their use case.
    """

    # External method assignments
    _create_sentencepiece_vocab = _create_sentencepiece_vocab
    _train_sentencepiece_model = _train_sentencepiece_model
    _create_frequency_vocab = _create_frequency_vocab
    _analyze_corpus = _analyze_corpus
    _select_vocabulary = _select_vocabulary
    _create_subword_vocab = _create_subword_vocab
    _optimize_token_ids = _optimize_token_ids
    _merge_vocabularies = _merge_vocabularies
    _merge_corpora = _merge_corpora
    _create_vocabulary_mappings = _create_vocabulary_mappings
    validate_pack = validate_pack
    compare_packs = compare_packs
    _get_pack_version = _get_pack_version
    _save_pack = _save_pack
    _cleanup_temp_files = _cleanup_temp_files

    def __init__(
        self,
        corpus_dir: str = "",
        output_dir: str = "",
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
        self.runtime_dirs = RuntimeDirectoryManager()
        self.corpus_dir = Path(corpus_dir) if corpus_dir else self.runtime_dirs.corpus_dir
        self.output_dir = Path(output_dir) if output_dir else self.runtime_dirs.vocab_dir
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
                languages=['en', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'sv', 'pl', 'id', 'vi', 'tr'],
                description='Latin script languages',
                recommended_mode=CreationMode.PRODUCTION
            ),
            'cjk': LanguageGroup(
                name='cjk',
                languages=['zh', 'ja', 'ko'],
                description='Chinese, Japanese, Korean',
                recommended_mode=CreationMode.PRODUCTION
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
            'thai': LanguageGroup(
                name='thai',
                languages=['th'],
                description='Thai script languages',
                recommended_mode=CreationMode.PRODUCTION
            ),
            'research': LanguageGroup(
                name='research',
                languages=['en'],
                description='Research and experimentation',
                recommended_mode=CreationMode.RESEARCH
            )
        }

    def _build_fingerprint(
        self,
        mode: Optional[CreationMode] = None,
        groups_to_create: Optional[List[str]] = None,
    ) -> str:
        """SHA-256 fingerprint of config + corpus state to detect changes."""
        h = hashlib.sha256()
        h.update(str(self.config.vocab_size).encode())
        h.update(str(self.config.vocab_type).encode())
        h.update(str(mode).encode() if mode else b'')
        h.update(str(sorted(groups_to_create or list(self.language_groups.keys()))).encode())
        for lang in sorted(self.corpus_paths):
            path = Path(self.corpus_paths[lang])
            if path.exists():
                s = path.stat()
                h.update(f"{lang}:{s.st_size}:{s.st_mtime_ns}".encode())
        return h.hexdigest()

    def _fingerprint_path(self) -> Path:
        return self.output_dir / '.vocab_fingerprint.json'

    def _skip_unchanged(
        self,
        mode: Optional[CreationMode] = None,
        groups_to_create: Optional[List[str]] = None,
    ) -> bool:
        """Return True if vocab is already built with this exact config + corpus."""
        fp_path = self._fingerprint_path()
        if not fp_path.exists():
            return False
        try:
            old = json.loads(fp_path.read_text())
        except Exception:
            return False
        cur = self._build_fingerprint(mode, groups_to_create)
        return old.get('fingerprint') == cur

    def _save_fingerprint(
        self,
        mode: Optional[CreationMode] = None,
        groups_to_create: Optional[List[str]] = None,
    ) -> None:
        """Persist fingerprint so future runs can skip."""
        fp = self._build_fingerprint(mode, groups_to_create)
        self._fingerprint_path().write_text(json.dumps({
            'fingerprint': fp,
            'vocab_size': self.config.vocab_size,
            'groups': sorted(groups_to_create or list(self.language_groups.keys())),
        }, indent=2))
        logger.debug(f"Saved vocab fingerprint: {fp[:12]}...")

    def create_all_packs(
        self,
        mode: Optional[CreationMode] = None,
        groups_to_create: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Create all or specified vocabulary packs.

        Skips if existing packs match the current config + corpus fingerprint,
        so re-running after the pipeline (or with identical settings) is a no-op.

        Args:
            mode: Override creation mode (None = use recommended)
            groups_to_create: List of group names (None = all)

        Returns:
            Dictionary of created packs
        """
        # ── Skip if unchanged ──────────────────────────────────────
        if self._skip_unchanged(mode, groups_to_create):
            logger.info("Vocabulary packs are already up-to-date (config and corpus unchanged).")
            return {}

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

        # Save fingerprint so future identical runs skip
        if created_packs:
            self._save_fingerprint(mode, groups_to_create)

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
                vocab_data = self._evolve_pack(
                    pack_name, base_pack_path, tokens_to_add, mode
                )
            else:
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

            # Copy SentencePiece model with versioned name
            version = pack['version']
            model_src = self.output_dir / f"{pack_name}.model"
            model_dst = self.output_dir / f"{pack_name}_v{version}.model"
            if model_src.exists() and not model_dst.exists():
                import shutil
                shutil.copy2(model_src, model_dst)
                logger.info(f"Saved SentencePiece model to {model_dst}")

            logger.info(f"Successfully created pack '{pack_name}'")
            self._log_pack_stats(pack)

            return pack

        except Exception as e:
            logger.error(f"Failed to create pack {pack_name}: {e}")
            raise VocabularyError(f"Pack creation failed: {e}") from e

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

        # Determine bump type from creation mode
        bump = "major" if mode == CreationMode.RESEARCH else "minor"
        version = self._get_pack_version(pack_name, bump=bump)

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
                    'compatible_decoder': '>=1.0.0',
                'model_file': f'{pack_name}_v{version}.model',
            'hash': hashlib.sha256(json.dumps(vocab_data, sort_keys=True).encode()).hexdigest()[:16],
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Unified Vocabulary Pack Creator")
    parser.add_argument("command", choices=["create", "create_all"], help="Operation to perform")
    parser.add_argument("--pack", dest="pack_name", help="Pack name (for create)")
    parser.add_argument("--langs", nargs="*", help="Languages (for create)")
    parser.add_argument("--mode", choices=["production", "research", "hybrid"], default="production")
    parser.add_argument("--corpus_dir", default=str(self.runtime_dirs.processed_dir))
    parser.add_argument("--output_dir", default=str(RuntimeDirectoryManager().vocab_dir))
    parser.add_argument("--groups", nargs="*", help="Groups to create for create_all")
    args = parser.parse_args()

    from enum import Enum
    class _M(Enum):
        PRODUCTION = "production"; RESEARCH = "research"; HYBRID = "hybrid"
    mode_map = {"production": CreationMode.PRODUCTION, "research": CreationMode.RESEARCH, "hybrid": CreationMode.HYBRID}

    creator = UnifiedVocabularyCreator(corpus_dir=args.corpus_dir, output_dir=args.output_dir, default_mode=mode_map[args.mode])
    if args.command == "create":
        if not args.pack_name or not args.langs:
            raise SystemExit("--pack and --langs are required for 'create'")
        creator.create_pack(pack_name=args.pack_name, languages=args.langs, mode=mode_map[args.mode])
    else:
        creator.create_all_packs(mode=mode_map[args.mode], groups_to_create=args.groups)
