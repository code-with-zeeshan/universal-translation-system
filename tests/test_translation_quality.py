# tests/test_translation_quality.py
import pytest
import numpy as np
from typing import List, Tuple

class TranslationQualityTester:
    def __init__(self):
        self.test_pairs = self._load_test_pairs()
        self.bleu_scorer = BLEUScorer()
        self.bert_scorer = BERTScorer()
        
    def test_translation_quality(self, encoder_sdk, decoder_url):
        """Test translation quality across all language pairs"""
        results = []
        
        for source_lang, target_lang, test_sentences in self.test_pairs:
            for source, reference in test_sentences:
                # Translate
                translation = self._translate(
                    source, source_lang, target_lang, 
                    encoder_sdk, decoder_url
                )
                
                # Score
                bleu = self.bleu_scorer.score(reference, translation)
                bert = self.bert_scorer.score(reference, translation)
                
                results.append({
                    'pair': f"{source_lang}-{target_lang}",
                    'bleu': bleu,
                    'bert': bert,
                    'acceptable': bleu > 0.3 and bert > 0.8
                })
                
        return results
    
    def benchmark_performance(self, encoder_sdk):
        """Benchmark encoder performance on device"""
        metrics = {
            'latency': [],
            'memory': [],
            'cpu_usage': []
        }
        
        for text_length in [10, 50, 100, 200]:
            text = " ".join(["word"] * text_length)
            
            # Measure latency
            start = time.perf_counter()
            encoded = encoder_sdk.encode(text, "en", "es")
            latency = (time.perf_counter() - start) * 1000
            
            metrics['latency'].append(latency)
            
        return metrics