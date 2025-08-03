# encoder/adapter_composition.py

import torch
import torch.nn as nn
from collections import OrderedDict
from typing import List, Dict

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

    # You can add more advanced strategies here in the future,
    # such as weighted averaging or task-vector arithmetic.
    # @staticmethod
    # def weighted_average(adapters: List[nn.Module], weights: List[float]) -> OrderedDict:
    #     ...