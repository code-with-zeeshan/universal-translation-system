import torch
from torch.utils.data import Sampler, Dataset
from collections import defaultdict
import random
import numpy as np
from typing import List, Dict, Iterator, Optional, Union
import logging

logger = logging.getLogger(__name__)


class TemperatureSampler(Sampler[Union[int, List[int]]]):
    """
    Samples elements from multiple language groups with temperature scaling.
    Supports both index-per-item and batch outputs.

    Args:
        data_source: Dataset with language pair info (via get_lang_pair_indices() or __getitem__ metadata).
        batch_size: Batch size for training.
        temperature: T > 1.0 flattens distribution, T < 1.0 sharpens, T = 1.0 is proportional.
        output_batches: If True, yields lists of indices (batches). If False, yields single indices.
        drop_last: Whether to drop incomplete last batch.
    """
    def __init__(self, data_source: Dataset, batch_size: int, temperature: float = 1.0,
                 output_batches: bool = True, drop_last: bool = False):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.temperature = temperature
        self.output_batches = output_batches
        self.drop_last = drop_last

        self.lang_pair_indices = self._get_lang_pair_indices()
        self.lang_pairs = list(self.lang_pair_indices.keys())

        base_probs = torch.tensor([len(indices) for indices in self.lang_pair_indices.values()], dtype=torch.float)

        if self.temperature != 1.0:
            scaled_logits = torch.log(base_probs) / self.temperature
            self.sampling_weights = torch.softmax(scaled_logits, dim=0)
        else:
            self.sampling_weights = base_probs / base_probs.sum()

        self.num_samples = len(self.data_source)
        if self.drop_last and self.num_samples % self.batch_size != 0:
            self.num_batches = self.num_samples // self.batch_size
        else:
            self.num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size

    def _get_lang_pair_indices(self) -> Dict[str, List[int]]:
        """Get language pair indices from dataset."""
        if hasattr(self.data_source, 'get_lang_pair_indices'):
            return self.data_source.get_lang_pair_indices()

        indices = defaultdict(list)
        for i in range(len(self.data_source)):
            item = self.data_source[i]
            if isinstance(item, dict) and 'metadata' in item:
                meta = item['metadata']
                pair = f"{meta['source_lang']}-{meta['target_lang']}"
            else:
                pair = "default"
            indices[pair].append(i)
        return indices

    def __iter__(self) -> Iterator[Union[int, List[int]]]:
        if self.output_batches:
            return self._iter_batches()
        return self._iter_individual()

    def _iter_individual(self) -> Iterator[int]:
        num_samples_per_pair = torch.multinomial(self.sampling_weights, self.num_samples, replacement=True)
        epoch_indices = []
        for i, pair in enumerate(self.lang_pairs):
            count = num_samples_per_pair[i].item()
            indices = self.lang_pair_indices[pair]
            sampled = np.random.choice(indices, size=count, replace=len(indices) < count)
            epoch_indices.extend(sampled.tolist())
        np.random.shuffle(epoch_indices)
        return iter(epoch_indices)

    def _iter_batches(self) -> Iterator[List[int]]:
        for _ in range(self.num_batches):
            batch_indices = []
            for _ in range(self.batch_size):
                chosen_pair = self.lang_pairs[torch.multinomial(self.sampling_weights, 1).item()]
                batch_indices.append(random.choice(self.lang_pair_indices[chosen_pair]))
            yield batch_indices

    def __len__(self) -> int:
        if self.output_batches:
            return self.num_batches
        return self.num_samples


class BalancedLanguageSampler(Sampler):
    """
    Additional sampler for strict balanced sampling across language pairs.
    Ensures each language pair gets equal representation.
    """

    def __init__(self, dataset: Dataset, batch_size: int):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size

        if not hasattr(dataset, 'get_lang_pair_indices'):
            raise ValueError("Dataset must have 'get_lang_pair_indices' method")

        self.lang_pair_indices = dataset.get_lang_pair_indices()
        self.lang_pairs = list(self.lang_pair_indices.keys())

        # Create balanced sampling order
        self._create_balanced_order()

    def _create_balanced_order(self):
        """Create balanced sampling order"""
        # Interleave samples from each language pair
        iterators = {
            pair: iter(random.sample(indices, len(indices)))
            for pair, indices in self.lang_pair_indices.items()
        }

        self.sampling_order = []
        exhausted = set()

        while len(exhausted) < len(self.lang_pairs):
            for pair in self.lang_pairs:
                if pair not in exhausted:
                    try:
                        idx = next(iterators[pair])
                        self.sampling_order.append(idx)
                    except StopIteration:
                        exhausted.add(pair)

    def __iter__(self):
        """Generate samples in balanced order"""
        for i in range(0, len(self.sampling_order), self.batch_size):
            batch = self.sampling_order[i:i + self.batch_size]
            if batch:
                yield batch

    def __len__(self):
        """Return number of batches"""
        return (len(self.sampling_order) + self.batch_size - 1) // self.batch_size
