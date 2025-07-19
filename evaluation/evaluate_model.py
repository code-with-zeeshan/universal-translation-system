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

# Metrics imports
try:
    from sacrebleu import corpus_bleu, sentence_bleu, BLEU
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False
    print("Warning: sacrebleu not available. Install with: pip install sacrebleu")

try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    print("Warning: COMET not available. Install with: pip install unbabel-comet")

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
        """Tokenize text using vocabulary pack"""
        # Simple whitespace tokenization - replace with proper tokenizer
        tokens = [vocab_pack.special_tokens.get('<s>', 2)]  # Start token
        
        for word in text.lower().split():
            if word in vocab_pack.tokens:
                tokens.append(vocab_pack.tokens[word])
            else:
                # Handle unknown words with subwords
                tokens.append(vocab_pack.special_tokens.get('<unk>', 1))
        
        tokens.append(vocab_pack.special_tokens.get('</s>', 3))  # End token
        return tokens
    
    def _detokenize(self, token_ids: np.ndarray, vocab_pack) -> str:
        """Convert token IDs back to text"""
        # Create reverse mapping
        id_to_token = {v: k for k, v in vocab_pack.tokens.items()}
        id_to_token.update({v: k for k, v in vocab_pack.special_tokens.items()})
        
        tokens = []
        for token_id in token_ids:
            if token_id == vocab_pack.special_tokens.get('</s>', 3):
                break
            if token_id == vocab_pack.special_tokens.get('<pad>', 0):
                continue
            
            token = id_to_token.get(int(token_id), '<unk>')
            if not token.startswith('<'):  # Skip special tokens
                tokens.append(token)
        
        # Join and clean up
        text = ' '.join(tokens)
        text = text.replace(' ##', '')  # Handle subwords
        text = text.replace('▁', ' ')   # Handle sentencepiece tokens
        
        return text.strip()
    
    def evaluate_dataset(self, 
                        test_data: List[TranslationPair],
                        batch_size: int = 32,
                        use_cache: bool = True) -> Dict[str, Any]:
        """
        Evaluate translation quality on a dataset
        
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
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        for metric, value in report['summary'].items():
            print(f"{metric.upper()}: {value:.4f}")
        print("="*50)
        
        return report


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
    print("Translation Evaluation Module")
    print("This module requires initialized encoder/decoder models and test data.")
    print("\nExample usage:")
    print("evaluator = TranslationEvaluator(encoder, decoder, vocab_manager)")
    print("metrics = evaluator.evaluate_file('test_data.tsv')")