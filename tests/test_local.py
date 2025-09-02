import pytest
import torch
from encoder.universal_encoder import UniversalEncoder


def test_encoder_forward_shape_cpu():
    encoder = UniversalEncoder()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    attention_mask = torch.ones_like(input_ids)
    out = encoder(input_ids=input_ids, attention_mask=attention_mask)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 5, encoder.hidden_dim)