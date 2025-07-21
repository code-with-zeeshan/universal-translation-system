import pytest
import torch
from cloud_decoder.optimized_decoder import OptimizedUniversalDecoder

def test_decoder_basic():
    decoder = OptimizedUniversalDecoder()
    encoder_hidden = torch.randn(1, 10, 1024)  # batch, seq, encoder_dim
    decoder_input_ids = torch.randint(0, 100, (1, 10))
    output = decoder.forward(
        decoder_input_ids=decoder_input_ids,
        encoder_hidden_states=encoder_hidden
    )
    assert output.shape[0] == 1  # batch size
    assert isinstance(output, torch.Tensor)