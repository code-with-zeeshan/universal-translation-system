import pytest
import torch
from encoder.universal_encoder import UniversalEncoder
from cloud_decoder.optimized_decoder import OptimizedUniversalDecoder

def test_translation_pipeline():
    # Initialize encoder
    encoder = UniversalEncoder()
    encoder.load_vocabulary_pack({'tokens': {'<bos>': 0, '<eos>': 2, '<unk>': 1, 'hello': 3, 'how': 4, 'are': 5, 'you': 6, '?': 7}})
    text = "Hello, how are you?"
    input_ids = torch.tensor([[3, 4, 5, 6, 7]])
    encoded = encoder(input_ids)
    assert encoded.shape[0] == 1
    # Initialize decoder
    decoder = OptimizedUniversalDecoder()
    decoder_input_ids = torch.tensor([[3, 4, 5, 6, 7]])
    output = decoder.forward(
        decoder_input_ids=decoder_input_ids,
        encoder_hidden_states=encoded
    )
    assert output.shape[0] == 1
    assert isinstance(output, torch.Tensor)