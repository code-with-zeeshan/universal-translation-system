import pytest
import torch
from encoder.universal_encoder import UniversalEncoder

def test_encoder_basic():
    encoder = UniversalEncoder()
    input_ids = torch.tensor([[1, 2, 3, 4]])  # Example token IDs
    # Provide attention_mask to align with updated forward signature
    attention_mask = torch.ones_like(input_ids)
    output = encoder(input_ids=input_ids, attention_mask=attention_mask)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 4, encoder.hidden_dim)
