# evaluation/evaluate_model.py
"""
Comprehensive evaluation module for the Universal Translation System
Supports BLEU, COMET, and custom metrics
"""

import torch
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import logging
from dataclasses import dataclass
from collections import defaultdict
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("Warning: pandas not available. Install with: pip install pandas")

# Metrics imports
try:
    from sacrebleu import corpus_bleu, sentence_bleu, BLEU
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False
    logger.warning("Warning: sacrebleu not available. Install with: pip install sacrebleu")

try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    logger.warning("Warning: COMET not available. Install with: pip install unbabel-comet")

logger = logging.getLogger(__name__)


@dataclass
class TranslationPair:
    """Single translation pair for evaluation"""
    source: str
    target: str
    source_lang: str
    target_lang: str
    predicted: Optional[str] = None


class TranslationEvaluator:
    """Comprehensive translation quality evaluator"""
    
    def __init__(self, 
                 encoder_model: torch.nn.Module,
                 decoder_model: torch.nn.Module,
                 vocabulary_manager,
                 device: str = 'cuda',
                 comet_model_name: str = "Unbabel/wmt22-comet-da"):
        
        self.encoder = encoder_model
        self.decoder = decoder_model
        self.vocab_manager = vocabulary_manager
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Move models to device
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        
        # Set to eval mode
        self.encoder.eval()
        self.decoder.eval()
        
        # Load COMET model if available
        self.comet_model = None
        if COMET_AVAILABLE:
            try:
                model_path = download_model(comet_model_name)
                self.comet_model = load_from_checkpoint(model_path)
                logger.info(f"✅ Loaded COMET model: {comet_model_name}")
            except Exception as e:
                logger.warning(f"Failed to load COMET model: {e}")
        
        # Initialize BLEU scorer
        self.bleu_scorer = BLEU() if SACREBLEU_AVAILABLE else None
        
    def translate(self, 
                  source_text: str, 
                  source_lang: str, 
                  target_lang: str,
                  max_length: int = 128) -> str:
        """Translate a single text using the encoder-decoder system"""
        
        with torch.no_grad():
            # Get vocabulary pack
            vocab_pack = self.vocab_manager.get_vocab_for_pair(source_lang, target_lang)
            
            # Tokenize source text
            source_tokens = self._tokenize(source_text, source_lang, vocab_pack)
            source_ids = torch.tensor([source_tokens], device=self.device)
            attention_mask = torch.ones_like(source_ids)
            
            # Encode
            encoder_output = self.encoder(source_ids, attention_mask, language=source_lang)
            
            # Get target language ID
            target_lang_id = vocab_pack.special_tokens.get(f'<{target_lang}>', 2)
            
            # Decode
            generated_ids, _ = self.decoder.generate(
                encoder_hidden_states=encoder_output,
                encoder_attention_mask=attention_mask,
                target_lang_id=target_lang_id,
                max_length=max_length,
                temperature=0.7,
                top_k=50,
                top_p=0.9
            )
            
            # Convert tokens to text
            translation = self._detokenize(generated_ids[0].cpu().numpy(), vocab_pack)
            
            return translation
    
    def _tokenize(self, text: str, language: str, vocab_pack) -> List[int]:
        """Tokenize text using vocabulary pack with proper subword and better error handling"""
        try:
            # Check if vocab pack has a tokenize method
            if hasattr(vocab_pack, 'tokenize'):
                return vocab_pack.tokenize(text, language)

            # Otherwise, implement proper tokenization
            tokens = []
    
            # Add start token
            start_token = vocab_pack.special_tokens.get(f'<{language}>', vocab_pack.special_tokens.get('<s>', 2))
            tokens.append(start_token)
    
            # Tokenize text
            words = text.lower().split()
    
            for word in words:
                if word in vocab_pack.tokens:
                    # Word exists in vocabulary
                    tokens.append(vocab_pack.tokens[word])
                else:
                    # Try subword tokenization
                    subword_tokens = self._subword_tokenize(word, vocab_pack)
                    tokens.extend(subword_tokens)

            # Add end token 
            end_token = vocab_pack.special_tokens.get('</s>', 3)
            tokens.append(end_token)

            return tokens        

        except Exception as e:
            logger.error(f"Tokenization failed for '{text[:50]}...': {e}")
            # Return UNK tokens as fallback
            return [
                vocab_pack.special_tokens.get('<s>', 2),
                vocab_pack.special_tokens.get('<unk>', 1),
                vocab_pack.special_tokens.get('</s>', 3)
            ]

    def _subword_tokenize(self, word: str, vocab_pack) -> List[int]:
        """Subword tokenization for OOV words"""
        subword_tokens = []
    
        # First, check if the word with ## prefix exists
        if hasattr(vocab_pack, 'subwords'):
            # Try progressively smaller subwords
            for i in range(len(word)):
                for j in range(len(word), i, -1):
                    subword = word[i:j]
                
                    # First subword doesn't need ##
                    if i == 0:
                        if subword in vocab_pack.tokens:
                            subword_tokens.append(vocab_pack.tokens[subword])
                            i = j - 1
                            break
                    else:
                        # Subsequent subwords need ## prefix
                        subword_with_prefix = f"##{subword}"
                        if subword_with_prefix in vocab_pack.subwords:
                            subword_tokens.append(vocab_pack.subwords[subword_with_prefix])
                            i = j - 1
                            break
    
        # If no subwords found, use UNK token
        if not subword_tokens:
            subword_tokens.append(vocab_pack.special_tokens.get('<unk>', 1))
    
        return subword_tokens
    
    def _detokenize(self, token_ids: np.ndarray, vocab_pack) -> str:
        """Convert token IDs back to text with proper subword handling"""
        # Create reverse mapping
        id_to_token = {}
    
        # Add all token mappings
        for mapping_dict in [vocab_pack.tokens, vocab_pack.special_tokens]:
            id_to_token.update({v: k for k, v in mapping_dict.items()})
    
        if hasattr(vocab_pack, 'subwords'):
            id_to_token.update({v: k for k, v in vocab_pack.subwords.items()})
    
        tokens = []
        for token_id in token_ids:
            # Skip padding
            if token_id == vocab_pack.special_tokens.get('<pad>', 0):
                continue
            
            # Stop at end token
            if token_id == vocab_pack.special_tokens.get('</s>', 3):
                break
        
            token = id_to_token.get(int(token_id), '<unk>')
        
            # Skip special tokens
            if token.startswith('<') and token.endswith('>'):
                continue
            
            tokens.append(token)
    
        # Join tokens and handle subwords
        text = ''
        for i, token in enumerate(tokens):
            if token.startswith(' '):
                # SentencePiece style: leading space indicates a new word
                text += ' ' + token[1:]
            elif token.startswith('##'):
                # Subword: append without space
                text += token[2:]
            else:
                # Regular token: append with space if not first token
                if i > 0:
                    text += ' '
                text += token
    
        return text.strip()

    def evaluate_dataset_streaming(self, 
                                test_data_iterator,
                                max_samples: Optional[int] = None,
                                batch_size: int = 32,
                                cache_translations: bool = True) -> Dict[str, Any]:
        """
        Evaluate with streaming to handle large datasets efficiently
    
        Args:
            test_data_iterator: Iterator that yields TranslationPair objects
            max_samples: Maximum number of samples to evaluate
            batch_size: Batch size for processing
            cache_translations: Whether to cache translations
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"📊 Starting streaming evaluation...")
    
        predictions = []
        references = []
        source_texts = []
    
        # Streaming cache
        cache = {} if cache_translations else None
    
        # Batch accumulator
        batch_buffer = []
        sample_count = 0
    
        # Metrics accumulators for streaming calculation
        running_metrics = {
            'total_loss': 0.0,
            'total_correct': 0,
            'total_chars': 0,
            'correct_chars': 0
        }
    
        try:
            for pair in test_data_iterator:
                batch_buffer.append(pair)
            
                # Process when batch is full
                if len(batch_buffer) >= batch_size:
                    batch_results = self._process_batch(batch_buffer, cache)
                
                    # Update running metrics
                    self._update_streaming_metrics(batch_results, running_metrics)
                
                    # Store results for final calculation
                    predictions.extend(batch_results['predictions'])
                    references.extend(batch_results['references'])
                    source_texts.extend(batch_results['sources'])
                    
                    # Clear batch buffer
                    batch_buffer = []
                    sample_count += batch_size
                
                    # Log progress periodically
                    if sample_count % 1000 == 0:
                        interim_bleu = self._calculate_interim_bleu(
                            predictions[-1000:], 
                            references[-1000:]
                        )
                        logger.info(f"Processed {sample_count} samples, "
                                  f"recent BLEU: {interim_bleu:.2f}")
                
                    # Memory management
                    if sample_count % 5000 == 0:
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                    # Check max samples
                    if max_samples and sample_count >= max_samples:
                        break
        
            # Process remaining batch
            if batch_buffer:
                batch_results = self._process_batch(batch_buffer, cache)
                self._update_streaming_metrics(batch_results, running_metrics)
                predictions.extend(batch_results['predictions'])
                references.extend(batch_results['references'])
                source_texts.extend(batch_results['sources'])
                sample_count += len(batch_buffer)
        
            # Calculate final metrics
            logger.info(f"📊 Calculating final metrics on {sample_count} samples...")
        
            # For streaming, we might want to calculate metrics on a subset
            # to avoid memory issues with very large datasets
            if len(predictions) > 10000:
                # Sample for final metric calculation
                indices = np.random.choice(len(predictions), 10000, replace=False)
                sampled_predictions = [predictions[i] for i in indices]
                sampled_references = [references[i] for i in indices]
                sampled_sources = [source_texts[i] for i in indices]
            else:
                sampled_predictions = predictions
                sampled_references = references
                sampled_sources = source_texts
        
            metrics = self._calculate_metrics(sampled_predictions, sampled_references, sampled_sources)
        
            # Add streaming-specific metrics
            metrics['streaming_metrics'] = {
                'total_samples': sample_count,
                'avg_char_accuracy': running_metrics['correct_chars'] / max(running_metrics['total_chars'], 1)
            }
        
            # Save cache if enabled
            if cache and cache_translations:
                cache_file = Path("streaming_evaluation_cache.json")
                with open(cache_file, 'w') as f:
                    json.dump(cache, f, ensure_ascii=False, indent=2)
                logger.info(f"💾 Saved translation cache to {cache_file}")
        
            return metrics
        
        except Exception as e:
            logger.error(f"❌ Streaming evaluation failed: {e}")
            raise

    def _process_batch(self, batch: List[TranslationPair], cache: Optional[Dict] = None) -> Dict[str, List]:
        """Process a batch of translation pairs"""
        predictions = []
        references = []
        sources = []
    
        for pair in batch:
            # Check cache
            cache_key = f"{pair.source_lang}-{pair.target_lang}:{pair.source}"
        
            if cache and cache_key in cache:
                translation = cache[cache_key]
            else:
                # Translate
                translation = self.translate(
                    pair.source, 
                    pair.source_lang, 
                    pair.target_lang
                )
            
                if cache is not None:
                    cache[cache_key] = translation
        
            predictions.append(translation)
            references.append(pair.target)
            sources.append(pair.source)
    
        return {
            'predictions': predictions,
            'references': references,
            'sources': sources
        }

    def _update_streaming_metrics(self, batch_results: Dict, running_metrics: Dict):
        """Update running metrics for streaming evaluation"""
        for pred, ref in zip(batch_results['predictions'], batch_results['references']):
            # Character-level accuracy
            for c1, c2 in zip(pred, ref):
                running_metrics['total_chars'] += 1
                if c1 == c2:
                    running_metrics['correct_chars'] += 1

    def _calculate_interim_bleu(self, predictions: List[str], references: List[str]) -> float:
        """Calculate BLEU score for interim results"""
        if SACREBLEU_AVAILABLE:
            try:
                bleu = corpus_bleu(predictions, [references])
                return bleu.score
            except:
                return 0.0
        return 0.0

    # Adding streaming support to file evaluation:
    def evaluate_file_streaming(self, test_file: str, 
                               file_format: str = 'tsv',
                               chunk_size: int = 1000) -> Dict[str, Any]:
        """
        Evaluate from a test file using streaming
        """
        def file_iterator():
            """Generator that yields TranslationPair objects from file"""
            with open(test_file, 'r', encoding='utf-8') as f:
                for line_no, line in enumerate(f, 1):
                    if file_format == 'tsv':
                        parts = line.strip().split('\t')
                        if len(parts) >= 4:
                            yield TranslationPair(
                                source=parts[0],
                                target=parts[1],
                                source_lang=parts[2],
                                target_lang=parts[3]
                            )
                    elif file_format == 'json':
                        data = json.loads(line)
                        yield TranslationPair(**data)
                
                    # Log progress
                    if line_no % 10000 == 0:
                        logger.info(f"Read {line_no} lines from {test_file}")
    
        return self.evaluate_dataset_streaming(file_iterator())
    
    def evaluate_dataset(self, 
                        test_data: List[TranslationPair],
                        batch_size: int = 32,
                        use_cache: bool = True) -> Dict[str, Any]:
        """Evaluate translation quality on a dataset
        
        Args:
            test_data: List of TranslationPair objects
            batch_size: Batch size for translation
            use_cache: Whether to cache translations
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"📊 Evaluating on {len(test_data)} samples...")
        
        predictions = []
        references = []
        source_texts = []
        
        # Cache file
        cache_file = Path("evaluation_cache.json")
        cache = {}
        
        if use_cache and cache_file.exists():
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        
        # Process in batches
        for i in tqdm(range(0, len(test_data), batch_size), desc="Translating"):
            batch = test_data[i:i + batch_size]
            
            for pair in batch:
                # Check cache
                cache_key = f"{pair.source_lang}-{pair.target_lang}:{pair.source}"
                
                if use_cache and cache_key in cache:
                    translation = cache[cache_key]
                else:
                    # Translate
                    translation = self.translate(
                        pair.source, 
                        pair.source_lang, 
                        pair.target_lang
                    )
                    
                    if use_cache:
                        cache[cache_key] = translation
                
                pair.predicted = translation
                predictions.append(translation)
                references.append(pair.target)
                source_texts.append(pair.source)
        
        # Save cache
        if use_cache:
            with open(cache_file, 'w') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, references, source_texts)
        
        # Add language pair analysis
        metrics['language_pair_scores'] = self._analyze_by_language_pair(test_data)
        
        return metrics
    
    def _calculate_metrics(self, 
                          predictions: List[str], 
                          references: List[str],
                          sources: Optional[List[str]] = None) -> Dict[str, float]:
        """Calculate all available metrics"""
        metrics = {}
        
        # BLEU score
        if SACREBLEU_AVAILABLE and self.bleu_scorer:
            try:
                bleu_score = corpus_bleu(predictions, [references])
                metrics['bleu'] = bleu_score.score
                metrics['bleu_details'] = {
                    'score': bleu_score.score,
                    'counts': bleu_score.counts,
                    'totals': bleu_score.totals,
                    'precisions': bleu_score.precisions,
                    'bp': bleu_score.bp,
                    'sys_len': bleu_score.sys_len,
                    'ref_len': bleu_score.ref_len
                }
                logger.info(f"BLEU: {bleu_score.score:.2f}")
            except Exception as e:
                logger.error(f"BLEU calculation failed: {e}")
                metrics['bleu'] = 0.0
        
        # COMET score
        if COMET_AVAILABLE and self.comet_model and sources:
            try:
                # Prepare data for COMET
                comet_data = [
                    {
                        "src": src,
                        "mt": pred,
                        "ref": ref
                    }
                    for src, pred, ref in zip(sources, predictions, references)
                ]
                
                # Calculate COMET scores
                comet_scores = self.comet_model.predict(
                    comet_data, 
                    batch_size=32, 
                    gpus=1 if torch.cuda.is_available() else 0
                )
                
                metrics['comet'] = float(np.mean(comet_scores.scores))
                metrics['comet_details'] = {
                    'mean': float(np.mean(comet_scores.scores)),
                    'std': float(np.std(comet_scores.scores)),
                    'min': float(np.min(comet_scores.scores)),
                    'max': float(np.max(comet_scores.scores))
                }
                logger.info(f"COMET: {metrics['comet']:.4f}")
            except Exception as e:
                logger.error(f"COMET calculation failed: {e}")
                metrics['comet'] = 0.0
        
        # Character-level accuracy
        char_accuracies = []
        for pred, ref in zip(predictions, references):
            if len(ref) > 0:
                correct_chars = sum(c1 == c2 for c1, c2 in zip(pred, ref))
                accuracy = correct_chars / max(len(pred), len(ref))
                char_accuracies.append(accuracy)
        
        metrics['char_accuracy'] = float(np.mean(char_accuracies))
        
        # Length ratio
        pred_lengths = [len(p.split()) for p in predictions]
        ref_lengths = [len(r.split()) for r in references]
        metrics['length_ratio'] = float(np.mean(pred_lengths)) / float(np.mean(ref_lengths))
        
        # Token-level metrics
        metrics['avg_pred_length'] = float(np.mean(pred_lengths))
        metrics['avg_ref_length'] = float(np.mean(ref_lengths))
        
        return metrics
    
    def _analyze_by_language_pair(self, test_data: List[TranslationPair]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by language pair"""
        pair_groups = defaultdict(list)
        
        # Group by language pair
        for pair in test_data:
            key = f"{pair.source_lang}-{pair.target_lang}"
            pair_groups[key].append(pair)
        
        # Calculate metrics for each pair
        pair_metrics = {}
        
        for pair_key, pairs in pair_groups.items():
            predictions = [p.predicted for p in pairs if p.predicted]
            references = [p.target for p in pairs]
            
            if predictions and references:
                if SACREBLEU_AVAILABLE:
                    bleu = corpus_bleu(predictions, [references])
                    pair_metrics[pair_key] = {
                        'bleu': bleu.score,
                        'sample_count': len(predictions)
                    }
                else:
                    pair_metrics[pair_key] = {
                        'sample_count': len(predictions)
                    }
        
        return pair_metrics
    
    def evaluate_file(self, test_file: str, file_format: str = 'tsv') -> Dict[str, Any]:
        """Evaluate from a test file"""
        test_data = self._load_test_file(test_file, file_format)
        return self.evaluate_dataset(test_data)
    
    def _load_test_file(self, file_path: str, file_format: str = 'tsv') -> List[TranslationPair]:
        """Load test data from file"""
        test_data = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if file_format == 'tsv':
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        test_data.append(TranslationPair(
                            source=parts[0],
                            target=parts[1],
                            source_lang=parts[2],
                            target_lang=parts[3]
                        ))
                elif file_format == 'json':
                    data = json.loads(line)
                    test_data.append(TranslationPair(**data))
        
        logger.info(f"Loaded {len(test_data)} test samples from {file_path}")
        return test_data
    
    def create_evaluation_report(self, metrics: Dict[str, Any], output_file: str = "evaluation_report.json"):
        """Create detailed evaluation report"""
        
        report = {
            "summary": {
                "bleu": metrics.get('bleu', 0.0),
                "comet": metrics.get('comet', 0.0),
                "char_accuracy": metrics.get('char_accuracy', 0.0),
                "length_ratio": metrics.get('length_ratio', 1.0)
            },
            "details": metrics,
            "language_pairs": metrics.get('language_pair_scores', {}),
            "metadata": {
                "evaluation_date": str(Path.ctime(Path.cwd())),
                "model_info": {
                    "encoder": self.encoder.__class__.__name__,
                    "decoder": self.decoder.__class__.__name__,
                    "device": str(self.device)
                }
            }
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Evaluation report saved to {output_file}")
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*50)
        for metric, value in report['summary'].items():
            logger.info(f"{metric.upper()}: {value:.4f}")
        logger.info("="*50)
        
        return report

    def compare_models(self, 
                       models: List[Tuple[torch.nn.Module, torch.nn.Module, str]],
                       test_data: List[TranslationPair]) -> Optional['pd.DataFrame']:
        """Compare multiple models on the same test set"""
        if not PANDAS_AVAILABLE:
            logger.warning("pandas not available for model comparison")
            return None
            
        import pandas as pd
    
        results = []
    
        for encoder, decoder, model_name in models:
            # Temporarily swap models
            old_encoder, old_decoder = self.encoder, self.decoder
            self.encoder, self.decoder = encoder, decoder
        
            # Evaluate
            metrics = self.evaluate_dataset(test_data)
            metrics['model'] = model_name
            results.append(metrics)
        
            # Restore original models
            self.encoder, self.decoder = old_encoder, old_decoder
    
        # Create comparison dataframe
        df = pd.DataFrame(results)
        return df 

    def profile_memory_usage(self, test_data: List[TranslationPair]) -> Dict[str, float]:
        """Profile memory usage during evaluation"""
        import tracemalloc
        import gc
    
        gc.collect()
        tracemalloc.start()
    
        # Get baseline
        baseline = tracemalloc.get_traced_memory()[0]
    
        # Run evaluation
        metrics = self.evaluate_dataset(test_data[:100])  # Small sample
    
        # Get peak memory
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    
        return {
            'baseline_mb': baseline / 1024 / 1024,
            'peak_mb': peak / 1024 / 1024,
            'increase_mb': (peak - baseline) / 1024 / 1024,
            'per_sample_kb': (peak - baseline) / 1024 / len(test_data[:100])
        }    

# Standalone evaluation function
def evaluate_translation_quality(encoder_model, decoder_model, vocab_manager, test_data_path: str) -> Dict[str, Any]:
    """Quick evaluation function for testing"""
    
    evaluator = TranslationEvaluator(
        encoder_model=encoder_model,
        decoder_model=decoder_model,
        vocabulary_manager=vocab_manager
    )
    
    # Load test data
    test_data = evaluator._load_test_file(test_data_path)
    
    # Evaluate
    metrics = evaluator.evaluate_dataset(test_data)
    
    # Create report
    report = evaluator.create_evaluation_report(metrics)
    
    return report


if __name__ == "__main__":
    # Example usage
    logger.info("Translation Evaluation Module")
    logger.info("This module requires initialized encoder/decoder models and test data.")
    logger.info("\nExample usage:")
    logger.info("evaluator = TranslationEvaluator(encoder, decoder, vocab_manager)")
    logger.info("metrics = evaluator.evaluate_file('test_data.tsv')")")}``)