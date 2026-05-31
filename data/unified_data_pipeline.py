# data/unified_data_pipeline.py
"""
Unified data pipeline manager combining all data management functionality.
Replaces: data_manager.py and practical_data_pipeline.py
"""

import logging
import os
import random
from typing import Iterator

from torch.utils.data import Sampler, Dataset

from utils.logging_config import setup_logging
from data.pipeline_state import PipelineStage, PipelineState
from data.pipeline_orchestrator import UnifiedDataPipeline

# Ensure centralized logging for data pipeline
setup_logging(log_dir="logs", log_level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("data")


# ============= CUSTOM SAMPLERS (from data_manager.py) =============
# TemperatureSampler is in data/custom_samplers.py

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


__all__ = [
    "PipelineStage",
    "PipelineState",
    "BalancedLanguageSampler",
    "UnifiedDataPipeline",
]


def main():
    """CLI entry point for the unified data pipeline"""
    from data.pipeline_orchestrator import main as orch_main
    orch_main()


if __name__ == "__main__":
    main()
