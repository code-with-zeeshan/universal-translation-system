// scripts/convert-model.js
const fs = require('fs');
const path = require('path');

console.log(`
===========================================
ONNX Model Preparation for Web
===========================================

This script helps prepare your model for web deployment.

Requirements:
1. Python with ONNX installed (pip install onnx onnxruntime)
2. Your PyTorch model file

Steps to convert your model:

1. Export from PyTorch to ONNX:
   python -c "
   import torch
   from encoder.universal_encoder import UniversalEncoder
   
   # Load your model
   model = UniversalEncoder()
   model.load_state_dict(torch.load('models/universal_encoder.pt'))
   model.eval()
   
   # Create dummy input
   dummy_input = torch.randint(0, 1000, (1, 128))
   dummy_mask = torch.ones(1, 128, dtype=torch.long)
   
   # Export to ONNX
   torch.onnx.export(
       model,
       (dummy_input, dummy_mask),
       'models/universal_encoder.onnx',
       input_names=['input_ids', 'attention_mask'],
       output_names=['encoder_output'],
       dynamic_axes={
           'input_ids': {0: 'batch', 1: 'sequence'},
           'attention_mask': {0: 'batch', 1: 'sequence'},
           'encoder_output': {0: 'batch', 1: 'sequence'}
       },
       opset_version=14,
       do_constant_folding=True
   )
   "

2. Optimize for web (optional but recommended):
   python -m onnxruntime.tools.optimizer_cli \
     --input models/universal_encoder.onnx \
     --output models/universal_encoder_optimized.onnx \
     --optimization_level 99

3. Copy the model to the web SDK:
   cp models/universal_encoder_optimized.onnx web/universal-translation-sdk/public/models/universal_encoder.onnx

Note: INT8 quantized models may not work well in browsers. 
Consider using FP16 or FP32 for web deployment.
`);

// Check if model exists
const modelPath = path.join(__dirname, '../public/models/universal_encoder.onnx');
if (fs.existsSync(modelPath)) {
  const stats = fs.statSync(modelPath);
  console.log(`\n✅ Model found at: ${modelPath}`);
  console.log(`   Size: ${(stats.size / 1024 / 1024).toFixed(2)} MB`);
} else {
  console.log(`\n❌ Model not found at: ${modelPath}`);
  console.log('   Please follow the steps above to convert your model.');
}