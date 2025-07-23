# universal-decoder-node/tests/test_decoder.py
import pytest
import numpy as np
import torch
from universal_decoder_node.decoder import OptimizedUniversalDecoder, DecoderService


def test_decoder_initialization():
    """Test decoder model initialization"""
    model = OptimizedUniversalDecoder()
    assert model is not None
    assert model.decoder_dim == 512
    assert model.vocab_size == 50000


def test_decoder_forward():
    """Test decoder forward pass"""
    model = OptimizedUniversalDecoder()
    model.eval()
    
    batch_size = 2
    seq_len = 10
    encoder_dim = 1024
    
    # Create dummy inputs
    decoder_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    encoder_hidden_states = torch.randn(batch_size, seq_len, encoder_dim)
    
    # Forward pass
    with torch.no_grad():
        output = model(decoder_input_ids, encoder_hidden_states)
    
    assert output.shape == (batch_size, seq_len, model.vocab_size)


@pytest.mark.asyncio
async def test_decoder_service():
    """Test decoder service initialization"""
    service = DecoderService()
    assert service is not None
    assert service.app is not None


def test_compression():
    """Test embedding compression/decompression"""
    from universal_decoder_node.decoder import DecoderService
    
    service = DecoderService()
    
    # Create test data
    hidden_states = np.random.randn(1, 10, 1024).astype(np.float32)
    scale = 127.0
    
    # Compress
    import lz4.frame
    quantized = (hidden_states * scale).astype(np.int8)
    metadata = bytearray()
    metadata.extend(hidden_states.shape[1].to_bytes(4, 'little'))
    metadata.extend(hidden_states.shape[2].to_bytes(4, 'little'))
    metadata.extend(np.float32(scale).tobytes())
    
    compressed = lz4.frame.compress(quantized.tobytes())
    compressed_data = bytes(metadata) + compressed
    
    # Decompress
    result = service._decompress_encoder_output(compressed_data)
    
    assert 'hidden_states' in result
    assert result['hidden_states'].shape == hidden_states.shape