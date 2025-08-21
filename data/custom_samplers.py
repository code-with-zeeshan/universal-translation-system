# data/custom_samplers.py
import torch
from torch.utils.data import Sampler
from collections import defaultdict
import numpy as np
from typing import List, Dict, Iterator
import logging

logger = logging.getLogger(__name__)

class TemperatureSampler(Sampler[int]):
    """
    Samples elements from multiple language groups with temperature scaling.

    Args:
        data_source: The dataset to sample from. Must have a 'get_lang_pair' method
                     or a way to access language pair info for each index.
        batch_size: The size of batches to generate.
        temperature: The temperature for sampling. T > 1.0 flattens the distribution,
                     T < 1.0 sharpens it. T = 1.0 is standard proportional sampling.
    """
    def __init__(self, data_source, batch_size: int, temperature: float = 1.0):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.temperature = temperature
        
        # Get language pair counts from the dataset
        self.lang_pair_indices = self._get_lang_pair_indices()
        self.lang_pairs = list(self.lang_pair_indices.keys())
        
        # Calculate base probabilities
        base_probs = torch.tensor([len(indices) for indices in self.lang_pair_indices.values()], dtype=torch.float)
        
        # Apply temperature scaling
        if self.temperature != 1.0:
            # Use log-space for numerical stability
            scaled_logits = torch.log(base_probs) / self.temperature
            self.sampling_weights = torch.softmax(scaled_logits, dim=0)
        else:
            self.sampling_weights = base_probs / base_probs.sum()

        self.num_samples = len(self.data_source)
        
        logger.info(f"TemperatureSampler initialized with T={self.temperature}")
        logger.info(f"Sampling weights: {dict(zip(self.lang_pairs, self.sampling_weights.tolist()))}")

    def _get_lang_pair_indices(self) -> Dict[str, List[int]]:
        """Group indices by language pair."""
        indices = defaultdict(list)
        for i in range(len(self.data_source)):
            # This requires your dataset's __getitem__ to return metadata
            # which your ModernParallelDataset already does!
            metadata = self.data_source[i]['metadata']
            pair = f"{metadata['source_lang']}-{metadata['target_lang']}"
            indices[pair].append(i)
        return indices

    def __iter__(self) -> Iterator[int]:
        """
        Yields a batch of indices at a time.
        """
        # Determine the number of samples to draw from each language pair for the whole epoch
        num_samples_per_pair = torch.multinomial(self.sampling_weights, self.num_samples, replacement=True)
        
        # Create the full list of indices for the epoch
        epoch_indices = []
        for i, pair in enumerate(self.lang_pairs):
            count = num_samples_per_pair[i].item()
            indices = self.lang_pair_indices[pair]
            # Sample with replacement if necessary
            sampled_indices = np.random.choice(indices, size=count, replace=len(indices) < count)
            epoch_indices.extend(sampled_indices)

        # Shuffle the combined list of indices
        np.random.shuffle(epoch_indices)
        
        return iter(epoch_indices)

    def __len__(self) -> int:
        return self.num_samples