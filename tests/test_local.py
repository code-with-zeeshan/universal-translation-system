# test_local.py
import sys
sys.path.append('.')

from encoder_core.universal_encoder import UniversalEncoder
from cloud_decoder.optimized_decoder import OptimizedUniversalDecoder

def test_translation():
    print("🧪 Testing Universal Translation System")
    
    # Initialize encoder
    encoder = UniversalEncoder("models/production/encoder.onnx")
    encoder.load_vocabulary("vocabulary/latin_v1.msgpack")
    
    # Test encoding
    text = "Hello, how are you?"
    encoded = encoder.encode(text, "en", "es")
    print(f"✅ Encoded: {len(encoded)} bytes")
    
    # Initialize decoder
    decoder = OptimizedUniversalDecoder()
    decoder.load_state_dict(torch.load("models/production/decoder.pt"))
    
    # Test decoding
    translation = decoder.decode(encoded, "es")
    print(f"✅ Translation: {translation}")

if __name__ == "__main__":
    test_translation()