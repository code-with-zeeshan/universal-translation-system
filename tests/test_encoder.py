import pytest
import torch
from encoder.universal_encoder import UniversalEncoder

def test_encoder_basic():
    encoder = UniversalEncoder()
    input_ids = torch.tensor([[1, 2, 3, 4]])  # Example token IDs
    output = encoder(input_ids)
    assert output.shape[0] == 1  # batch size
    assert output.shape[2] == encoder.hidden_dim
    assert isinstance(output, torch.Tensor)
