# data/custom_samplers.py

import torch
from torch.utils.data import Sampler
from collections import defaultdict
import numpy as np
from typing import List, Dict, Iterator

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
        
        print(f"TemperatureSampler initialized with T={self.temperature}")
        print(f"Sampling weights: {dict(zip(self.lang_pairs, self.sampling_weights.tolist()))}")

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
        # Create a list of indices for each language pair to sample from
        lang_pair_iters = {
            pair: iter(np.random.permutation(indices))
            for pair, indices in self.lang_pair_indices.items()
        }

        # Generate batches until we've yielded enough samples
        for _ in range(len(self)):
            # Choose a language pair based on the temperature-scaled weights
            chosen_pair = np.random.choice(self.lang_pairs, p=self.sampling_weights.numpy())
            
            # Get the next index from that language pair's iterator
            try:
                index = next(lang_pair_iters[chosen_pair])
            except StopIteration:
                # If we've exhausted a language pair, reshuffle and create a new iterator
                lang_pair_iters[chosen_pair] = iter(np.random.permutation(self.lang_pair_indices[chosen_pair]))
                index = next(lang_pair_iters[chosen_pair])
            
            yield index

    def __len__(self) -> int:
        return self.num_samples