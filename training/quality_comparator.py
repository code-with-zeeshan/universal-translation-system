# training/quality_comparator.py
"""
A/B Testing for different quality levels
"""

import torch
import time
import logging
from typing import Dict, Any

from training.quantization_common import fake_quantize_tensor

logger = logging.getLogger(__name__)


class QualityComparator:
    """A/B Testing for different quality levels"""

    def __init__(self):
        """Initialize with quantization_aware flag"""
        self.quantization_aware = False  # Add this property

    def enable_quantization_aware_training(self):
        """Enable QAT mode"""
        self.quantization_aware = True

    def compare_models(self, text: str, source_lang: str, target_lang: str,
                      models: Dict[str, torch.nn.Module]) -> Dict[str, Dict[str, Any]]:
        """Compare translation quality across different model versions"""

        results = {}

        for model_name, model in models.items():
            start_time = time.time()

            # Get translation
            translation = self._translate(model, text, source_lang, target_lang)

            # Calculate metrics
            latency = (time.time() - start_time) * 1000  # ms

            results[model_name] = {
                'translation': translation,
                'latency_ms': latency,
                'model_size_mb': self._get_model_size_mb(model)
            }

        # Calculate relative quality metrics
        if 'fp32' in results:
            reference = results['fp32']['translation']
            for model_name in results:
                if model_name != 'fp32':
                    results[model_name]['similarity_score'] = self._calculate_similarity(
                        reference, results[model_name]['translation']
                    )

        return results

    def _translate(self, model: torch.nn.Module, text: str,
              source_lang: str, target_lang: str) -> str:
        """Perform actual translation using the encoder/decoder pipeline"""
        # Properly handle model structure
        if isinstance(model, tuple):
           encoder, decoder = model
        elif hasattr(model, 'encoder') and hasattr(model, 'decoder'):
            encoder = model.encoder
            decoder = model.decoder
        else:
            logger.error("Model structure not recognized")
            return ""

        # Import vocabulary manager
        from vocabulary.unified_vocab_manager import UnifiedVocabularyManager, VocabularyMode
        vocab_manager = UnifiedVocabularyManager(mode=VocabularyMode.OPTIMIZED)
        vocab_pack = vocab_manager.get_vocab_for_pair(source_lang, target_lang)

        # Tokenize input
        tokens = []
        tokens.append(vocab_pack.special_tokens.get('<s>', 2))

        # Simple tokenization (replace with production tokenizer)
        for word in text.lower().split():
            if word in vocab_pack.tokens:
               tokens.append(vocab_pack.tokens[word])
            else:
                # Handle unknown words with subwords or UNK
                tokens.append(vocab_pack.special_tokens.get('<unk>', 1))

        tokens.append(vocab_pack.special_tokens.get('</s>', 3))

        # Convert to tensor
        input_ids = torch.tensor([tokens], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        # Forward pass
        with torch.no_grad():
            encoder_output = encoder(input_ids, attention_mask)

            # Simple greedy decoding (replace with beam search in production)
            decoder_input = torch.tensor([[vocab_pack.special_tokens.get('<s>', 2)]], dtype=torch.long)
            max_length = 128

            generated_tokens = []
            for _ in range(max_length):
                decoder_output = decoder(
                    decoder_input,
                    encoder_output,
                    encoder_attention_mask=attention_mask
                )

                # Get next token
                next_token = decoder_output[:, -1, :].argmax(dim=-1)
                generated_tokens.append(next_token.item())

                # Stop if EOS
                if next_token.item() == vocab_pack.special_tokens.get('</s>', 3):
                    break

                # Append to decoder input
                decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=1)

        # Detokenize
        id_to_token = {v: k for k, v in vocab_pack.tokens.items()}
        id_to_token.update({v: k for k, v in vocab_pack.special_tokens.items()})

        translation_tokens = []
        for token_id in generated_tokens:
            if token_id in id_to_token:
                token = id_to_token[token_id]
                if not token.startswith('<') and not token.endswith('>'):
                    translation_tokens.append(token)

        return ' '.join(translation_tokens)

    # Implementing the fake_quantize method properly:
    def fake_quantize(self, tensor: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
        """Implement fake quantization for QAT"""
        if not self.quantization_aware:
            return tensor
        return fake_quantize_tensor(tensor, num_bits)

    def _calculate_bleu(self, references, translations):
        import sacrebleu
        return sacrebleu.corpus_bleu(translations, [references]).score

    def _calculate_accuracy(self, references, translations):
        # Simple accuracy: exact match
        return sum(r == t for r, t in zip(references, translations)) / len(references)

    def _calculate_perplexity(self, model, data_loader):
        import numpy as np
        model.eval()
        losses = []
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids']
                labels = batch['labels']
                outputs = model(input_ids)
                loss = loss_fct(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                losses.append(loss.item())
        return float(np.exp(np.mean(losses)))

    def _calculate_similarity(self, ref: str, hyp: str) -> float:
        """Calculate similarity between translations"""
        # Simple character-level similarity for demo
        # In production, use BLEU, METEOR, or BERTScore
        from difflib import SequenceMatcher
        return SequenceMatcher(None, ref, hyp).ratio()

    def _get_model_size_mb(self, model: torch.nn.Module) -> float:
        """Calculate model size in MB"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        return param_size / (1024 * 1024)
