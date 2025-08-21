# vocabulary/create_vocabulary_packs_from_data.py
"""
Production-ready vocabulary pack creator using SentencePiece.

This module provides a robust, production-ready solution for creating vocabulary
packs for different language groups using Google's SentencePiece library.
It includes comprehensive error handling, logging, and modern Python practices.

Usage:
    creator = VocabularyPackCreator()
    creator.create_all_packs()
    
    # Or create specific pack
    creator.create_pack('latin', ['en', 'es', 'fr'])

    # Or custom configuration
    config = SentencePieceConfig(vocab_size=30000, num_threads=8)
    creator = VocabularyPackCreator(config=config)
"""

import json
import logging
import os
from utils.exceptions import DataError , VocabularyError
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import msgpack
import sentencepiece as smp
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SentencePieceConfig:
    """Configuration for SentencePiece training."""
    vocab_size: int = 25000
    model_type: str = 'bpe'
    character_coverage: float = 0.9995
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3
    split_by_whitespace: bool = True
    num_threads: int = 16
    max_sentence_length: int = 4192
    shuffle_input_sentence: bool = True


@dataclass
class LanguageGroup:
    """Definition of a language group for vocabulary pack creation."""
    name: str
    languages: List[str]
    description: str


@dataclass
class PackStats:
    """Statistics for vocabulary pack."""
    total_tokens: int
    size_mb: float
    languages_processed: int
    corpus_lines: int


class VocabularyPackCreator:
    """
    Production-ready vocabulary pack creator using SentencePiece.
    Can create new packs or evolve existing ones.
    
    This class provides a robust solution for creating vocabulary packs using
    Google's SentencePiece library with comprehensive error handling and logging.
    
    Attributes:
        corpus_dir: Directory containing corpus files
        output_dir: Directory for output files
        config: SentencePiece configuration
        language_groups: Predefined language groupings
    """
    
    def __init__(
        self,
        corpus_dir: str = 'data/processed',
        output_dir: str = 'vocabs',
        config: Optional[SentencePieceConfig] = None
    ) -> None:
        """
        Initialize the vocabulary pack creator.
        
        Args:
            corpus_dir: Directory containing corpus files
            output_dir: Directory for output vocabulary packs
            config: SentencePiece configuration (uses default if None)
        """
        self.corpus_dir = Path(corpus_dir)
        self.output_dir = Path(output_dir)
        self.config = config or SentencePieceConfig()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define language groups
        self.language_groups = self._define_language_groups()
        
        logger.info(f"Initialized VocabularyPackCreator")
        logger.info(f"Corpus directory: {self.corpus_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Available language groups: {list(self.language_groups.keys())}")
    
    def create_all_packs(self) -> Dict[str, Dict[str, Any]]:
        """
        Create all predefined vocabulary packs.
        
        Returns:
            Dictionary mapping pack names to pack data
            
        Raises:
            VocabularyError: If any pack creation fails critically
        """
        logger.info("Creating all vocabulary packs...")
        
        created_packs = {}
        failed_packs = []
        
        for group_name, group_info in self.language_groups.items():
            try:
                logger.info(f"\n📦 Creating {group_name} vocabulary pack...")
                logger.info(f"Languages: {group_info.languages}")
                logger.info(f"Description: {group_info.description}")
                
                pack = self.create_pack(group_name, group_info.languages)
                created_packs[group_name] = pack
                
            except Exception as e:
                logger.error(f"Failed to create {group_name} pack: {e}")
                failed_packs.append(group_name)
        
        # Summary
        logger.info(f"\n📊 Summary:")
        logger.info(f"Successfully created: {len(created_packs)} packs")
        logger.info(f"Failed: {len(failed_packs)} packs")
        
        if failed_packs:
            logger.warning(f"Failed packs: {failed_packs}")
        
        return created_packs
    
    def create_pack(
        self, 
        pack_name: str, 
        languages: List[str],
        creation_mode: str = 'production', # 'production' or 'research'
        base_pack_path: Optional[str] = None, # For evolution
        tokens_to_add: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create or evolve a vocabulary pack.
        - If base_pack_path is provided, it evolves the existing pack.
        - Otherwise, it creates a new one from scratch.
        
        Args:
            pack_name: Name for the vocabulary pack
            languages: List of language codes
            creation_mode: The method to use for vocab creation.
            
        Returns:
            Dictionary containing the vocabulary pack data
            
        Raises:
            VocabularyError: If invalid parameters provided
            VocabularyError: If pack creation fails
        """
        if not pack_name:
            raise VocabularyError("pack_name cannot be empty")

        try:
            if base_pack_path:
                # --- EVOLUTION MODE ---
                logger.info(f"Evolving existing pack '{pack_name}' from: {base_pack_path}")
                with open(base_pack_path, 'rb') as f:
                    # Use strict_map_key=False for compatibility
                    base_pack = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
                
                # Inherit properties from the base pack
                vocab_data = {
                    'tokens': base_pack.get('tokens', {}),
                    'subwords': base_pack.get('subwords', {}),
                    'special_tokens': base_pack.get('special_tokens', {})
                }
                languages = base_pack.get('languages', languages)
                
                # Add new tokens if provided
                if tokens_to_add:
                    # Find the next available token ID
                    all_ids = [id_ for d in vocab_data.values() for id_ in d.values()]
                    max_id = max(all_ids) if all_ids else -1
                    
                    newly_added_count = 0
                    for token in tokens_to_add:
                        # Ensure the token doesn't already exist in any category
                        if token not in vocab_data['tokens'] and \
                           token not in vocab_data['subwords'] and \
                           token not in vocab_data['special_tokens']:
                            max_id += 1
                            vocab_data['tokens'][token] = max_id
                            newly_added_count += 1
                    logger.info(f"Promoted {newly_added_count} new tokens to the vocabulary.")

            else:
                # --- CREATION MODE (existing logic) ---
                logger.info(f"Creating new vocabulary pack: {pack_name}")
                if not languages:
                    raise VocabularyError("languages list cannot be empty for new pack creation")
        
                
                logger.info(f"Languages: {languages}")
                logger.info(f"Base pack path: {base_pack_path}")
                logger.info(f"Tokens to add: {tokens_to_add}")

                # 1. Validate and merge corpora
                merged_corpus = self._merge_corpora(languages, pack_name)

                if creation_mode == 'production':
                    # 2. Train SentencePiece model
                    model_path = self._train_sentencepiece_model(merged_corpus, pack_name)
                    # 3. Create vocabulary mappings
                    vocab_data = self._create_vocabulary_mappings(model_path, languages)
                    # 4. Cleanup temporary files
                    self._cleanup_temp_files(merged_corpus, model_path)
                elif creation_mode == 'research':
                    # Implement the logic from tools/create_vocabulary_packs.py here
                    vocab_data = self._create_vocab_from_frequency_analysis(merged_corpus)
                else:
                    raise VocabularyError(f"Unknown creation_mode: {creation_mode}")

            # 5. Create pack structure
            pack = self._create_pack_structure(pack_name, languages, vocab_data)
            
            # 6. Save pack
            self._save_pack(pack, pack_name)
            
            logger.info(f"✅ Successfully created/evolved pack '{pack_name}' (version {pack['version']})")
            logger.info(f"📊 Total tokens: {pack['metadata']['total_tokens']:,}, Size: {pack['metadata']['size_mb']:.1f}MB")
            
            return pack
            
        except Exception as e:
            logger.error(f"Failed to create/evolve vocabulary pack {pack_name}: {e}")
            raise VocabularyError(f"Pack operation failed: {e}") from e
    
    def _define_language_groups(self) -> Dict[str, LanguageGroup]:
        """Define predefined language groups."""
        return {
            'latin': LanguageGroup(
                name='latin',
                languages=['en', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'sv', 'pl', 'id', 'vi', 'tr'],
                description='Latin script languages including Romance, Germanic, and others'
            ),
            'cjk': LanguageGroup(
                name='cjk',
                languages=['zh', 'ja', 'ko'],
                description='Chinese, Japanese, and Korean languages'
            ),
            'arabic': LanguageGroup(
                name='arabic',
                languages=['ar'],
                description='Arabic script languages'
            ),
            'devanagari': LanguageGroup(
                name='devanagari',
                languages=['hi'],
                description='Devanagari script languages (Hindi, etc.)'
            ),
            'cyrillic': LanguageGroup(
                name='cyrillic',
                languages=['ru', 'uk'],
                description='Cyrillic script languages'
            ),
            'thai': LanguageGroup(
                name='thai',
                languages=['th'],
                description='Thai language'
            )
        }
    
    def _merge_corpora(self, languages: List[str], pack_name: str) -> str:
        """
        Merge corpora from multiple languages into a single file.
        
        Args:
            languages: List of language codes
            pack_name: Name for temporary merged file
            
        Returns:
            Path to merged corpus file
            
        Raises:
            DataError: If corpus files are missing
        """
        # Create temporary merged corpus file
        merged_corpus = self.output_dir / f"temp_{pack_name}_corpus.txt"
        
        available_languages = []
        total_lines = 0
        
        with open(merged_corpus, 'w', encoding='utf-8') as out_file:
            for lang in languages:
                corpus_file = self.corpus_dir / f"{lang}_corpus.txt"
                
                if not corpus_file.exists():
                    logger.warning(f"⚠️  Missing corpus file for {lang}: {corpus_file}")
                    continue
                
                lang_lines = 0
                try:
                    with open(corpus_file, 'r', encoding='utf-8') as in_file:
                        for line in in_file:
                            line = line.strip()
                            if line:  # Skip empty lines
                                out_file.write(f"{line}\n")
                                lang_lines += 1
                    
                    available_languages.append(lang)
                    total_lines += lang_lines
                    logger.info(f"✅ Merged {lang_lines:,} lines from {lang}")
                    
                except IOError as e:
                    logger.error(f"❌ Error reading {corpus_file}: {e}")
                    raise
        
        if not available_languages:
            raise DataError(f"No corpus files found for languages: {languages}")
        
        logger.info(f"📄 Merged corpus: {total_lines:,} lines from {len(available_languages)} languages")
        return str(merged_corpus)
    
    def _train_sentencepiece_model(self, corpus_file: str, pack_name: str) -> str:
        """
        Train SentencePiece model on merged corpus.
        
        Args:
            corpus_file: Path to merged corpus file
            pack_name: Name for the model
            
        Returns:
            Path to trained model file
            
        Raises:
            VocabularyError: If training fails
        """
        model_prefix = str(self.output_dir / f"temp_{pack_name}")
        
        # Prepare training arguments
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
            f'--split_by_whitespace={self.config.split_by_whitespace}',
            f'--num_threads={self.config.num_threads}',
            f'--max_sentence_length={self.config.max_sentence_length}',
            f'--shuffle_input_sentence={self.config.shuffle_input_sentence}',
            f'--train_extremely_large_corpus=true',  # For large corpora
            f'--input_sentence_size=10000000',       # Limit sentences for memory
            f'--hard_vocab_limit=false',             # Allow flexible vocab size
            f'--byte_fallback=true',                 # Handle any Unicode
            f'--normalization_rule_name=identity',   # Preserve original text
        ]
        
        try:
            logger.info("🔧 Training SentencePiece model...")
            smp.SentencePieceTrainer.train(' '.join(train_args))
            
            model_path = f"{model_prefix}.model"
            if not Path(model_path).exists():
                raise VocabularyError("Model file not created")
            
            logger.info(f"✅ SentencePiece model trained: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"❌ SentencePiece training failed: {e}")
            raise VocabularyError(f"Model training failed: {e}") from e
    
    def _create_vocabulary_mappings(self, model_path: str, languages: List[str]) -> Dict[str, Any]:
        """
        Create vocabulary mappings from trained SentencePiece model.
        
        Args:
            model_path: Path to trained SentencePiece model
            languages: List of language codes
            
        Returns:
            Dictionary with vocabulary mappings
        """
        # Load trained model
        sp = smp.SentencePieceProcessor()
        sp.load(model_path)
        
        # Create vocabulary mappings
        tokens = {}
        subwords = {}
        special_tokens = {}
        
        # Process all pieces from the model
        for i in range(sp.get_piece_size()):
            piece = sp.id_to_piece(i)
            
            # Categorize tokens
            if piece in ['<pad>', '<unk>', '<s>', '</s>']:
                special_tokens[piece] = i
            elif piece.startswith('▁'):  # SentencePiece word boundary marker
                # Remove the boundary marker for cleaner token
                clean_piece = piece[1:] if len(piece) > 1 else piece
                tokens[clean_piece] = i
            elif piece.startswith('##'):  # Subword continuation
                subwords[piece] = i
            else:
                # Regular token or subword
                if len(piece) > 1 and not piece.isalnum():
                    subwords[piece] = i
                else:
                    tokens[piece] = i
        
        # Add language-specific tokens
        next_id = sp.get_piece_size()
        for lang in languages:
            lang_token = f'<{lang}>'
            if lang_token not in special_tokens:
                special_tokens[lang_token] = next_id
                next_id += 1
        
        # Add common special tokens
        common_special = ['<mask>', '<sep>', '<cls>']
        for token in common_special:
            if token not in special_tokens:
                special_tokens[token] = next_id
                next_id += 1
        
        logger.info(f"📊 Vocabulary created:")
        logger.info(f"  - Tokens: {len(tokens):,}")
        logger.info(f"  - Subwords: {len(subwords):,}")
        logger.info(f"  - Special tokens: {len(special_tokens):,}")
        logger.info(f"  - Total: {len(tokens) + len(subwords) + len(special_tokens):,}")
        
        return {
            'tokens': tokens,
            'subwords': subwords,
            'special_tokens': special_tokens
        }
    
    def _create_pack_structure(
        self, 
        pack_name: str, 
        languages: List[str], 
        vocab_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create the complete vocabulary pack structure.
        
        Args:
            pack_name: Name of the vocabulary pack
            languages: List of language codes
            vocab_data: Vocabulary mappings
            
        Returns:
            Complete vocabulary pack dictionary
        """
        # Calculate statistics
        total_tokens = (
            len(vocab_data['tokens']) + 
            len(vocab_data['subwords']) + 
            len(vocab_data['special_tokens'])
        )
        
        # Calculate size in MB
        packed_data = msgpack.packb(vocab_data)
        size_mb = len(packed_data) / (1024 * 1024)

        # Get dynamic version
        version = self._get_pack_version(pack_name)

        # Extract embeddings from trained model
        embeddings = self.extract_embeddings_for_tokens(tokens)
        
        # Create pack structure
        pack = {
            'name': pack_name,
            'version': version,
            'languages': languages,
            'tokens': vocab_data['tokens'],
            'embeddings': embeddings,  # Crucial for quality!
            'compression': 'int8',  # How embeddings are stored
            'subwords': vocab_data['subwords'],
            'special_tokens': vocab_data['special_tokens'],
            'metadata': {
                'total_tokens': total_tokens,
                'size_mb': size_mb,
                'vocab_size': self.config.vocab_size,
                'model_type': self.config.model_type,
                'character_coverage': self.config.character_coverage,
                'config': self.config.__dict__
            }
        }
        
        return pack
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), retry=retry_if_exception_type(IOError))
    def _save_pack(self, pack: Dict[str, Any], pack_name: str) -> None:
        """
        Save vocabulary pack in multiple formats with retry logic.
        
        Args:
            pack: Vocabulary pack data
            pack_name: Name for the pack files
        """
        try:
            # Use version from pack data
            version = pack.get('version', '1.0')

            # Save JSON format with version
            json_path = self.output_dir / f'{pack_name}_v{version}.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(pack, f, ensure_ascii=False, indent=2)
            
            # Save MessagePack format with version
            msgpack_path = self.output_dir / f'{pack_name}_v{version}.msgpack'
            with open(msgpack_path, 'wb') as f:
                f.write(msgpack.packb(pack))
            
            # Validate the saved pack
            is_valid, errors = self.validate_pack(str(json_path))
            if not is_valid:
                logger.warning(f"⚠️  Pack validation warnings: {errors}")

            logger.info(f"💾 Saved vocabulary pack:")
            logger.info(f"  - JSON: {json_path}")
            logger.info(f"  - MessagePack: {msgpack_path}")
            
        except IOError as e:
            logger.error(f"❌ Error saving pack: {e}")
            raise VocabularyError(f"Failed to save pack: {e}") from e
    
    def _cleanup_temp_files(self, merged_corpus: str, model_path: str) -> None:
        """
        Clean up temporary files created during processing.
        
        Args:
            merged_corpus: Path to temporary merged corpus file
            model_path: Path to temporary model file
        """
        try:
            # Clean up merged corpus
            if Path(merged_corpus).exists():
                os.remove(merged_corpus)
                logger.info(f"🧹 Cleaned up: {merged_corpus}")
            
            # Clean up model files
            model_prefix = model_path.replace('.model', '')
            for ext in ['.model', '.vocab']:
                temp_file = f"{model_prefix}{ext}"
                if Path(temp_file).exists():
                    os.remove(temp_file)
                    logger.info(f"🧹 Cleaned up: {temp_file}")
                    
        except OSError as e:
            logger.warning(f"⚠️  Could not clean up temporary files: {e}")

    def _get_pack_version(self, pack_name: str) -> str:
        """
        Generate version based on existing packs.
        
        Args:
            pack_name: Name of the pack
            
        Returns:
            Version string (e.g., "1.0", "1.1", "2.0")
        """
        existing_versions = []
        
        # Check for existing pack files
        for file in self.output_dir.glob(f'{pack_name}_v*.json'):
            # Extract version from filename (e.g., "latin_v1.2.json" -> "1.2")
            version_part = file.stem.split('_v')[-1]
            try:
                existing_versions.append(version_part)
            except VocabularyError:
                continue
        
        if not existing_versions:
            return "1.0"
        
        # Find the latest version and increment
        latest = sorted(existing_versions, key=lambda v: [int(x) for x in v.split('.')])[-1]
        major, minor = map(int, latest.split('.'))
        
        # Increment minor version by default
        return f"{major}.{minor + 1}" 

    def extract_embeddings_for_tokens(self, tokens: Dict[str, int]) -> Dict[str, List[float]]:
        """
        Extract embeddings for tokens from the trained model with fallback options.
        This is crucial for maintaining quality in quantized models.
        Args:
            tokens: Dictionary mapping tokens to IDs
        Returns:
            Dictionary mapping tokens to embedding vectors
        """
        import torch
        import os
        import numpy as np

        embeddings = {}
        embedding_dim = 768  # Default dimension
    
        # Try multiple approaches in order of preference

        # 1. Try to load custom trained model
        model_path = os.environ.get('ENCODER_MODEL_PATH', 'models/production/encoder.pt')
        fallback_model_path = os.environ.get('FALLBACK_MODEL_PATH', 'models/fallback/encoder.pt')
        embedding_dim = int(os.environ.get('EMBEDDING_DIM', '768'))
        if os.path.exists(model_path):
            try:
                logger.info(f"Loading embeddings from trained model: {model_path}")
            
                # Load model checkpoint
                checkpoint = torch.load(model_path, map_location='cpu')
            
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    # Assume it's the model itself
                    state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
            
                # Find embedding layer
                embedding_weight = None
                for key, tensor in state_dict.items():
                    if 'embedding' in key.lower() and 'weight' in key:
                        embedding_weight = tensor
                        embedding_dim = tensor.shape[1]
                        logger.info(f"Found embeddings: {key} with shape {tensor.shape}")
                        break
            
                if embedding_weight is not None:
                    # Extract embeddings for our tokens
                    for token, token_id in tokens.items():
                        if token_id < embedding_weight.shape[0]:
                            embeddings[token] = embedding_weight[token_id].tolist()
                        else:
                            # Initialize randomly for out-of-range tokens
                            embeddings[token] = (torch.randn(embedding_dim) * 0.02).tolist()
                
                    logger.info(f"✅ Extracted embeddings from trained model for {len(embeddings)} tokens")
                    return embeddings
                
            except Exception as e:
                logger.warning(f"Failed to load from trained model: {e}")
    
        # 2. Fallback to pre-trained sentence transformer
        try:
            logger.info("Using pre-trained sentence transformer as fallback...")
            from sentence_transformers import SentenceTransformer
        
            # Use multilingual model
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            embedding_dim = model.get_sentence_embedding_dimension()
        
            # Get embeddings for tokens
            token_list = list(tokens.keys())
            token_embeddings = model.encode(token_list, convert_to_tensor=True, show_progress_bar=True)
        
            for token, embedding in zip(token_list, token_embeddings):
                embeddings[token] = embedding.cpu().numpy().tolist()
        
            logger.info(f"✅ Generated embeddings using sentence transformer for {len(embeddings)} tokens")
            return embeddings
        
        except ImportError:
            logger.warning("sentence-transformers not available")
        except Exception as e:
            logger.warning(f"Failed to use sentence transformer: {e}")
    
        # 3. Fallback to random initialization with linguistic features
        logger.warning("Using random initialization with linguistic features as final fallback...")
    
        # Initialize with slight variations based on token properties
        for token, token_id in tokens.items():
            # Create base random embedding
            base_embedding = np.random.randn(embedding_dim) * 0.02
        
            # Add linguistic features
            if token.startswith('##'):  # Subword token
                base_embedding[0] = -0.5  # Mark as subword
            elif token.startswith('<') and token.endswith('>'):  # Special token
                base_embedding[1] = 1.0  # Mark as special
            elif len(token) == 1:  # Single character
                base_embedding[2] = 0.5  # Mark as single char
            
            # Add some hash-based consistency
            token_hash = hash(token) % 1000 / 1000.0
            base_embedding[3] = token_hash
        
            embeddings[token] = base_embedding.tolist()
    
        logger.info(f"✅ Initialized {len(embeddings)} embeddings with linguistic features")
    
        # Validate embeddings
        if embeddings:
            sample_token = list(embeddings.keys())[0]
            actual_dim = len(embeddings[sample_token])
            logger.info(f"Embedding dimension: {actual_dim}")
        
            # Ensure all embeddings have same dimension
            for token in embeddings:
                if len(embeddings[token]) != actual_dim:
                    logger.warning(f"Inconsistent embedding dimension for token '{token}'")
                    embeddings[token] = embeddings[token][:actual_dim] + [0.0] * (actual_dim - len(embeddings[token]))
    
        return embeddings
     
    def get_pack_info(self, pack_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about an existing vocabulary pack.
        
        Args:
            pack_name: Name of the vocabulary pack
            
        Returns:
            Pack information dictionary or None if not found
        """
        json_path = self.output_dir / f'{pack_name}_v1.json'
        
        if not json_path.exists():
            return None
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error reading pack info: {e}")
            return None
    
    def list_available_packs(self) -> List[str]:
        """
        List all available vocabulary packs.
        
        Returns:
            List of pack names
        """
        packs = []
        for json_file in self.output_dir.glob('*_v1.json'):
            pack_name = json_file.stem.replace('_v1', '')
            packs.append(pack_name)
        return packs

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
                if 'total_tokens' in pack['metadata']:
                    reported_total = pack['metadata']['total_tokens']
                    actual_total = (
                        len(pack.get('tokens', {})) + 
                        len(pack.get('subwords', {})) + 
                        len(pack.get('special_tokens', {}))
                    )
                    if reported_total != actual_total:
                        errors.append(f"Token count mismatch: reported {reported_total}, actual {actual_total}")
        
            return len(errors) == 0, errors
        
        except Exception as e:
            return False, [f"Error loading/parsing pack: {str(e)}"]


def main():
    """Example usage of the VocabularyPackCreator."""
    try:
        # Create vocabulary pack creator
        creator = VocabularyPackCreator(
            corpus_dir='data/processed',
            output_dir='vocabs'
        )
        
        # Create all packs
        logger.info("🚀 Starting vocabulary pack creation...")
        created_packs = creator.create_all_packs()
        
        # Display results
        logger.info(f"\n🎉 Process completed!")
        logger.info(f"Created {len(created_packs)} vocabulary packs:")
        
        for pack_name, pack_data in created_packs.items():
            metadata = pack_data.get('metadata', {})
            logger.info(f"  - {pack_name}: {metadata.get('total_tokens', 0):,} tokens, {metadata.get('size_mb', 0):.1f}MB")
        
        # List available packs
        available_packs = creator.list_available_packs()
        logger.info(f"\n📦 Available packs: {available_packs}")

        # Validate created packs
        logger.info("\n🔍 Validating created packs...")
        for pack_name in creator.list_available_packs():
            pack_path = f"vocabs/{pack_name}_v1.0.json"  # Adjust based on actual version
            is_valid, errors = creator.validate_pack(pack_path)
            if is_valid:
                logger.info(f"✅ {pack_name}: Valid")
            else:
                logger.warning(f"⚠️  {pack_name}: Invalid - {errors}")
        
    except Exception as e:
        logger.error(f"❌ Script execution failed: {e}")
        raise


if __name__ == "__main__":
    main()