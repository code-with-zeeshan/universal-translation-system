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
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential

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
            RuntimeError: If any pack creation fails critically
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
    
    def create_pack(self, pack_name: str, languages: List[str]) -> Dict[str, Any]:
        """
        Create vocabulary pack for specified languages.
        
        Args:
            pack_name: Name for the vocabulary pack
            languages: List of language codes
            
        Returns:
            Dictionary containing the vocabulary pack data
            
        Raises:
            ValueError: If invalid parameters provided
            RuntimeError: If pack creation fails
        """
        if not pack_name:
            raise ValueError("pack_name cannot be empty")
        if not languages:
            raise ValueError("languages list cannot be empty")
        
        logger.info(f"Creating vocabulary pack: {pack_name}")
        logger.info(f"Languages: {languages}")
        
        try:
            # 1. Validate and merge corpora
            merged_corpus = self._merge_corpora(languages, pack_name)
            
            # 2. Train SentencePiece model
            model_path = self._train_sentencepiece_model(merged_corpus, pack_name)
            
            # 3. Create vocabulary mappings
            vocab_data = self._create_vocabulary_mappings(model_path, languages)
            
            # 4. Create pack structure
            pack = self._create_pack_structure(pack_name, languages, vocab_data)
            
            # 5. Save pack
            self._save_pack(pack, pack_name)
            
            # 6. Cleanup temporary files
            self._cleanup_temp_files(merged_corpus, model_path)
            
            logger.info(f"✅ Successfully created {pack_name} pack")
            logger.info(f"📊 {len(vocab_data['tokens'])} tokens, {pack['metadata']['size_mb']:.1f}MB")
            
            return pack
            
        except Exception as e:
            logger.error(f"Failed to create vocabulary pack {pack_name}: {e}")
            raise RuntimeError(f"Pack creation failed: {e}") from e
    
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
            FileNotFoundError: If corpus files are missing
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
            raise FileNotFoundError(f"No corpus files found for languages: {languages}")
        
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
            RuntimeError: If training fails
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
            f'--shuffle_input_sentence={self.config.shuffle_input_sentence}'
        ]
        
        try:
            logger.info("🔧 Training SentencePiece model...")
            smp.SentencePieceTrainer.train(' '.join(train_args))
            
            model_path = f"{model_prefix}.model"
            if not Path(model_path).exists():
                raise RuntimeError("Model file not created")
            
            logger.info(f"✅ SentencePiece model trained: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"❌ SentencePiece training failed: {e}")
            raise RuntimeError(f"Model training failed: {e}") from e
    
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
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _save_pack(self, pack: Dict[str, Any], pack_name: str) -> None:
        """
        Save vocabulary pack in multiple formats.
        
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
            raise RuntimeError(f"Failed to save pack: {e}") from e
    
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
            except ValueError:
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
        Extract embeddings for tokens from the trained model.
        This is crucial for maintaining quality in quantized models.
        Args:
            tokens: Dictionary mapping tokens to IDs
        Returns:
            Dictionary mapping tokens to embedding vectors
        """
        import torch
        import os
        # Path to your trained encoder model (update as needed)
        model_path = os.environ.get('ENCODER_MODEL_PATH', 'models/production/encoder.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained encoder model not found at {model_path}. Set ENCODER_MODEL_PATH.")
        from encoder.universal_encoder import UniversalEncoder
        model = torch.load(model_path, map_location='cpu')
        if hasattr(model, 'embedding_layer'):
            embedding_weight = model.embedding_layer.weight.detach().cpu().numpy()
        else:
            raise AttributeError("Model does not have 'embedding_layer'. Update extraction logic.")
        embeddings = {}
        for token, token_id in tokens.items():
            if token_id < embedding_weight.shape[0]:
                embeddings[token] = embedding_weight[token_id].tolist()
            else:
                embeddings[token] = [0.0] * embedding_weight.shape[1]
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