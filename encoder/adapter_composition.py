# encoder/adapter_composition.py

import torch
import torch.nn as nn
from collections import OrderedDict
from typing import List, Dict, Optional, Union

class AdapterComposition:
    """
    Handles the logic for composing multiple language adapters.
    """
    @staticmethod
    def average_weights(adapters: List[nn.Module]) -> OrderedDict:
        """
        Averages the weights of a list of adapters.
        
        Args:
            adapters: A list of LanguageAdapter modules to be averaged.
            
        Returns:
            An OrderedDict containing the averaged state dictionary.
        """
        if not adapters:
            raise ValueError("Adapter list cannot be empty.")
            
        # Initialize with the state dict of the first adapter
        avg_state_dict = OrderedDict(adapters[0].state_dict())
        
        # Sum the weights from the other adapters
        for i in range(1, len(adapters)):
            for key in avg_state_dict.keys():
                avg_state_dict[key] += adapters[i].state_dict()[key]
        
        # Divide by the number of adapters to get the average
        num_adapters = len(adapters)
        for key in avg_state_dict.keys():
            avg_state_dict[key] = avg_state_dict[key] / num_adapters
            
        return avg_state_dict

    @staticmethod
    def weighted_average(adapters: List[nn.Module], weights: List[float]) -> OrderedDict:
        """
        Performs weighted averaging of adapter weights.
        
        Args:
            adapters: A list of LanguageAdapter modules to be averaged.
            weights: A list of weights for each adapter. Should sum to 1.0.
            
        Returns:
            An OrderedDict containing the weighted averaged state dictionary.
        """
        if not adapters:
            raise ValueError("Adapter list cannot be empty.")
        if len(adapters) != len(weights):
            raise ValueError("Number of adapters and weights must match.")
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")
            
        # Initialize with zeros
        weighted_state_dict = OrderedDict()
        for key in adapters[0].state_dict().keys():
            weighted_state_dict[key] = torch.zeros_like(adapters[0].state_dict()[key])
        
        # Add weighted contributions from each adapter
        for adapter, weight in zip(adapters, weights):
            for key in weighted_state_dict.keys():
                weighted_state_dict[key] += weight * adapter.state_dict()[key]
                
        return weighted_state_dict

    @staticmethod
    def task_vector_arithmetic(
        base_adapter: nn.Module,
        task_adapters: List[nn.Module],
        scaling_factors: Optional[List[float]] = None,
        operation: str = 'add'
    ) -> OrderedDict:
        """
        Performs task vector arithmetic for adapter composition.
        Task vectors are computed as the difference between fine-tuned and base adapters.
        
        Args:
            base_adapter: The base/reference adapter module.
            task_adapters: List of task-specific adapters.
            scaling_factors: Optional scaling factors for each task vector.
            operation: 'add' for addition, 'subtract' for subtraction.
            
        Returns:
            An OrderedDict containing the composed state dictionary.
        """
        if not task_adapters:
            raise ValueError("Task adapter list cannot be empty.")
        if operation not in ['add', 'subtract']:
            raise ValueError("Operation must be 'add' or 'subtract'.")
            
        if scaling_factors is None:
            scaling_factors = [1.0] * len(task_adapters)
        elif len(scaling_factors) != len(task_adapters):
            raise ValueError("Number of scaling factors must match number of task adapters.")
            
        # Start with base adapter weights
        result_state_dict = OrderedDict(base_adapter.state_dict())
        
        # Apply task vectors
        for task_adapter, scale in zip(task_adapters, scaling_factors):
            task_state_dict = task_adapter.state_dict()
            for key in result_state_dict.keys():
                # Compute task vector (difference from base)
                task_vector = task_state_dict[key] - base_adapter.state_dict()[key]
                
                # Apply operation
                if operation == 'add':
                    result_state_dict[key] += scale * task_vector
                else:  # subtract
                    result_state_dict[key] -= scale * task_vector
                    
        return result_state_dict

    @staticmethod
    def fisher_weighted_average(
        adapters: List[nn.Module],
        fisher_weights: List[Dict[str, torch.Tensor]]
    ) -> OrderedDict:
        """
        Performs Fisher information weighted averaging of adapters.
        Uses importance weights based on Fisher information matrix diagonal.
        
        Args:
            adapters: List of adapter modules.
            fisher_weights: List of dictionaries containing Fisher weights for each adapter.
            
        Returns:
            An OrderedDict containing the Fisher-weighted averaged state dictionary.
        """
        if not adapters:
            raise ValueError("Adapter list cannot be empty.")
        if len(adapters) != len(fisher_weights):
            raise ValueError("Number of adapters and Fisher weight dicts must match.")
            
        result_state_dict = OrderedDict()
        
        for key in adapters[0].state_dict().keys():
            # Collect all Fisher weights for this parameter
            all_fisher = torch.stack([fw[key] for fw in fisher_weights])
            
            # Normalize Fisher weights across adapters
            normalized_fisher = all_fisher / (all_fisher.sum(dim=0) + 1e-8)
            
            # Weighted average using normalized Fisher weights
            weighted_param = torch.zeros_like(adapters[0].state_dict()[key])
            for i, adapter in enumerate(adapters):
                weighted_param += normalized_fisher[i] * adapter.state_dict()[key]
                
            result_state_dict[key] = weighted_param
            
        return result_state_dict

    @staticmethod
    def regmean(
        adapters: List[nn.Module],
        lambda_reg: float = 0.1
    ) -> OrderedDict:
        """
        RegMean: Regularized mean for adapter merging.
        Averages adapters with L2 regularization to prevent overfitting.
        
        Args:
            adapters: List of adapter modules.
            lambda_reg: Regularization strength (default: 0.1).
            
        Returns:
            An OrderedDict containing the regularized mean state dictionary.
        """
        if not adapters:
            raise ValueError("Adapter list cannot be empty.")
            
        # First compute simple average
        avg_state_dict = AdapterComposition.average_weights(adapters)
        
        # Apply L2 regularization
        for key in avg_state_dict.keys():
            # Regularize towards zero (you could also regularize towards base model)
            avg_state_dict[key] = avg_state_dict[key] / (1 + lambda_reg)
            
        return avg_state_dict

    @staticmethod
    def ties_merging(
        adapters: List[nn.Module],
        threshold: float = 0.1,
        density: Optional[float] = None
    ) -> OrderedDict:
        """
        TIES (TrIm, Elect, and Merge) merging strategy.
        Trims small magnitude changes, resolves sign conflicts, and merges.
        
        Args:
            adapters: List of adapter modules.
            threshold: Magnitude threshold for trimming (default: 0.1).
            density: Optional target density for pruning.
            
        Returns:
            An OrderedDict containing the TIES-merged state dictionary.
        """
        if not adapters:
            raise ValueError("Adapter list cannot be empty.")
            
        result_state_dict = OrderedDict()
        
        for key in adapters[0].state_dict().keys():
            # Collect all parameter values
            params = torch.stack([adapter.state_dict()[key] for adapter in adapters])
            
            # Step 1: Trim - mask out small magnitude values
            if density is not None:
                # Use density-based pruning
                flat_params = params.abs().flatten()
                threshold_value = torch.quantile(flat_params, 1 - density)
                mask = params.abs() > threshold_value
            else:
                # Use threshold-based pruning
                mask = params.abs() > threshold
            
            masked_params = params * mask.float()
            
            # Step 2: Elect - resolve sign conflicts
            # Count positive and negative signs
            positive_counts = (masked_params > 0).sum(dim=0).float()
            negative_counts = (masked_params < 0).sum(dim=0).float()
            
            # Create sign election mask
            sign_mask = torch.sign(positive_counts - negative_counts)
            
            # Step 3: Merge - average aligned values
            aligned_params = masked_params * (torch.sign(masked_params) == sign_mask.unsqueeze(0)).float()
            counts = (aligned_params != 0).sum(dim=0).float()
            
            # Avoid division by zero
            counts = torch.where(counts == 0, torch.ones_like(counts), counts)
            merged_params = aligned_params.sum(dim=0) / counts
            
            result_state_dict[key] = merged_params
            
        return result_state_dict

    @staticmethod
    def dare_merging(
        adapters: List[nn.Module],
        drop_rate: float = 0.5,
        rescale: bool = True
    ) -> OrderedDict:
        """
        DARE (Drop And REscale) merging strategy.
        Randomly drops parameters and rescales remaining ones.
        
        Args:
            adapters: List of adapter modules.
            drop_rate: Probability of dropping each parameter (default: 0.5).
            rescale: Whether to rescale remaining parameters (default: True).
            
        Returns:
            An OrderedDict containing the DARE-merged state dictionary.
        """
        if not adapters:
            raise ValueError("Adapter list cannot be empty.")
        if not 0 <= drop_rate < 1:
            raise ValueError("Drop rate must be in [0, 1).")
            
        result_state_dict = OrderedDict()
        
        for key in adapters[0].state_dict().keys():
            # Initialize with zeros
            merged_param = torch.zeros_like(adapters[0].state_dict()[key])
            
            # Apply DARE to each adapter
            for adapter in adapters:
                param = adapter.state_dict()[key]
                
                # Create random mask
                mask = torch.bernoulli(torch.ones_like(param) * (1 - drop_rate))
                
                # Apply mask and optionally rescale
                if rescale and drop_rate > 0:
                    masked_param = param * mask / (1 - drop_rate)
                else:
                    masked_param = param * mask
                    
                merged_param += masked_param
            
            # Average across adapters
            result_state_dict[key] = merged_param / len(adapters)
            
        return result_state_dict

    @staticmethod
    def magnitude_pruning_merge(
        adapters: List[nn.Module],
        sparsity: float = 0.5
    ) -> OrderedDict:
        """
        Merges adapters by keeping only the highest magnitude parameters.
        
        Args:
            adapters: List of adapter modules.
            sparsity: Fraction of parameters to prune (default: 0.5).
            
        Returns:
            An OrderedDict containing the magnitude-pruned merged state dictionary.
        """
        if not adapters:
            raise ValueError("Adapter list cannot be empty.")
        if not 0 <= sparsity < 1:
            raise ValueError("Sparsity must be in [0, 1).")
            
        # First average the adapters
        avg_state_dict = AdapterComposition.average_weights(adapters)
        
        # Apply magnitude pruning
        for key in avg_state_dict.keys():
            param = avg_state_dict[key]
            
            # Calculate threshold for pruning
            flat_param = param.abs().flatten()
            k = int(sparsity * flat_param.numel())
            
            if k > 0:
                threshold = torch.topk(flat_param, k, largest=False)[0].max()
                mask = param.abs() > threshold
                avg_state_dict[key] = param * mask.float()
                
        return avg_state_dict

    @staticmethod
    def linear_combination(
        adapters: List[nn.Module],
        coefficients: List[float]
    ) -> OrderedDict:
        """
        Creates a linear combination of adapters with arbitrary coefficients.
        Unlike weighted_average, coefficients don't need to sum to 1.
        
        Args:
            adapters: List of adapter modules.
            coefficients: List of coefficients for linear combination.
            
        Returns:
            An OrderedDict containing the linear combination state dictionary.
        """
        if not adapters:
            raise ValueError("Adapter list cannot be empty.")
        if len(adapters) != len(coefficients):
            raise ValueError("Number of adapters and coefficients must match.")
            
        result_state_dict = OrderedDict()
        
        for key in adapters[0].state_dict().keys():
            combined_param = torch.zeros_like(adapters[0].state_dict()[key])
            
            for adapter, coeff in zip(adapters, coefficients):
                combined_param += coeff * adapter.state_dict()[key]
                
            result_state_dict[key] = combined_param
            
        return result_state_dict
    
# This Adapter Composition includes:

# 1. Weighted Average: Allows different importance weights for each adapter
# 2. Task Vector Arithmetic: Add/subtract task-specific adaptations from a base model
# 3. Fisher Weighted Average: Uses Fisher information for importance weighting
# 4. RegMean: Regularized averaging to prevent overfitting
# 5. TIES Merging: Advanced merging with trimming, sign election, and merging
# 6. DARE Merging: Drop and rescale strategy for robust merging
# 7. Magnitude Pruning: Keeps only high-magnitude parameters
# 8. Linear Combination: Arbitrary linear combinations without sum-to-1 constraint