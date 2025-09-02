import pytest
import torch
from cloud_decoder.optimized_decoder import OptimizedUniversalDecoder

def test_decoder_basic():
    decoder = OptimizedUniversalDecoder()
    encoder_hidden = torch.randn(1, 10, 1024)  # batch, seq, encoder_dim
    decoder_input_ids = torch.randint(0, 100, (1, 10))
    # Call forward without JIT annotation in tests for simplicity
    output = decoder(
        decoder_input_ids=decoder_input_ids,
        encoder_hidden_states=encoder_hidden
    )
    assert isinstance(output, torch.Tensor)
    assert output.dim() == 3  # [batch, seq, vocab]