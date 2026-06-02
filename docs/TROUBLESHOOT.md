# Comprehensive Troubleshooting Guide

## Table of Contents
1. [Installation Issues](#installation-issues)
2. [SDK Integration Issues](#sdk-integration-issues)
3. [Encoder Issues](#encoder-issues)
4. [Decoder Issues](#decoder-issues)
5. [Vocabulary Issues](#vocabulary-issues)
6. [Performance Issues](#performance-issues)
7. [Deployment Issues](#deployment-issues)
8. [Error Recovery Strategies](#error-recovery-strategies)
9. [Common Error Codes](#common-error-codes)

## Installation Issues

### Missing Dependencies
**Symptoms**: Import errors, module not found errors
**Solution**: 
```bash
# Use role-based install
bash scripts/install.sh --dev

# Or install modular requirements
pip install -r requirements/base.txt -r requirements/train.txt -r requirements/serve.txt
pip install -r requirements/decoder.txt -r requirements/coordinator.txt

# Check for missing dependencies
python scripts/check_dependencies.py
```

### CUDA Issues
**Symptoms**: "CUDA not available" warnings, CPU-only operation
**Solution**:
1. Verify CUDA: `nvidia-smi`
2. Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Reinstall PyTorch with CUDA: `pip install torch==2.0.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html`

## SDK Integration Issues

### Android SDK Issues
**Symptoms**: JNI errors, crashes on initialization
**Solutions**:
1. Check NDK version compatibility
2. Verify native library is properly included
3. See `sdk/android/UniversalTranslationSDK/` for details

### iOS SDK Issues
**Symptoms**: Missing symbols, crashes
**Solutions**:
1. Check framework linkage
2. See `sdk/ios/UniversalTranslationSDK/` for details

### Web SDK Issues
**Symptoms**: CORS errors, WASM loading failures
**Solutions**:
1. Ensure CORS properly configured
2. Check browser WASM compatibility
3. See `sdk/web/universal-translation-sdk/` for details

## Encoder Issues

### Encoding Failures
**Symptoms**: ENCODING_FAILED errors, empty embeddings
**Solutions**:
1. Check vocabulary availability
2. Verify input text format
3. Monitor memory usage

### Vocabulary Loading Failures
**Symptoms**: VOCABULARY_NOT_LOADED errors
**Solutions**:
1. Check vocabulary file existence
2. Verify format compatibility
3. See `vocabulary/` modules for creation utilities

## Decoder Issues

### Decoder Connection Issues
**Symptoms**: NETWORK_ERROR, connection timeouts
**Solutions**:
1. Check decoder health:
```bash
curl -v http://localhost:8001/health
curl -v http://localhost:5100/api/status
```
2. Implement retry logic:
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def translate_with_retry(client, text, source_lang, target_lang):
    return await client.translate(text, source_lang, target_lang)
```

### Decoder Performance Issues
**Symptoms**: Slow translations, timeouts
**Solutions**:
1. Check GPU utilization
2. Implement request batching
3. See `training/model_profiler.py` for profiling

## Vocabulary Issues

### Missing Vocabulary
**Symptoms**: VOCABULARY_NOT_LOADED errors
**Solutions**:
1. Check `vocabulary/vocab/` directory
2. Verify language code correctness
3. Create vocabulary packs:
```python
from vocabulary.vocabulary_creator import UnifiedVocabularyCreator, CreationMode

creator = UnifiedVocabularyCreator(corpus_dir='data/processed', output_dir='vocabs')
creator.create_pack(pack_name='latin', languages=['en','es','fr','de','it','pt','nl','sv','pl'], mode=CreationMode.PRODUCTION)

from vocabulary.unified_vocab_manager import UnifiedVocabularyManager, VocabularyMode
from config.schemas import RootConfig
manager = UnifiedVocabularyManager(config=RootConfig(), vocab_dir='vocabs', mode=VocabularyMode.OPTIMIZED)
pack = manager.get_vocab_for_pair(source_lang='en', target_lang='fr')
```

## Performance Issues

### Slow Translation
**Symptoms**: High latency, timeouts
**Solutions**:
1. Check network latency
2. Verify GPU utilization
3. Use profiling:
```python
from universal_decoder_node.utils.profiler import profile, profile_section, function_profiler

@profile
def my_translation_function():
    pass

bottlenecks = function_profiler.identify_bottlenecks()
```

### High Memory Usage
**Symptoms**: OOM errors, crashes
**Solutions**:
1. Reduce batch size
2. Use memory-efficient vocabulary loading
3. Use memory management system:
```python
from universal_decoder_node.utils.memory_manager import MemoryManager
memory_manager = MemoryManager.get_instance()
memory_stats = memory_manager.get_memory_stats()
memory_manager.cleanup()
```

## Deployment Issues

### Docker Deployment Issues
**Symptoms**: Container fails to start, crashes
**Solutions**:
1. Check Docker logs: `docker logs <container_id>`
2. Verify GPU: `docker exec <container_id> nvidia-smi`
3. Check volume mounts and permissions
4. Verify environment variables

### Kubernetes Deployment Issues
**Symptoms**: Pods fail to start, crash loops
**Solutions**:
1. Check pod logs: `kubectl logs <pod_name>`
2. Verify GPU: `kubectl exec <pod_name> -- nvidia-smi`
3. Check resource limits and requests
4. See `charts/uts/` for Helm deployment

## Error Recovery Strategies

### Circuit Breaker Pattern
The `CircuitBreaker` class in `coordinator/advanced_coordinator.py` handles failure detection:
```yaml
# Config
circuit_breaker:
  failure_threshold: 5
  recovery_timeout: 30
  timeout: 10
```

### Retry Logic
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def translate_with_retry(client, text, source_lang, target_lang):
    return await client.translate(text, source_lang, target_lang)
```

## Common Error Codes

| Error Code | Description | HTTP Status | Common Causes |
|------------|-------------|-------------|--------------|
| NETWORK_ERROR | Network connectivity issues | 503 | Connection failures, timeouts |
| MODEL_NOT_FOUND | Model file not found | 404 | Missing model files, incorrect paths |
| VOCABULARY_NOT_LOADED | Vocabulary not loaded | 404 | Missing vocabulary files |
| ENCODING_FAILED | Text encoding failed | 500 | Invalid input, memory issues |
| DECODING_FAILED | Embedding decoding failed | 500 | Invalid embeddings, model errors |
| INVALID_LANGUAGE | Unsupported language | 400 | Typos in language codes |
| RATE_LIMITED | Too many requests | 429 | Exceeding API limits |
| RESOURCE_EXHAUSTED | Out of resources | 503 | OOM, GPU memory exhausted |
| TIMEOUT | Request timed out | 504 | Slow network, overloaded servers |
| INVALID_INPUT | Invalid input parameters | 400 | Malformed requests |
| INTERNAL_ERROR | Unexpected internal error | 500 | Bugs, unhandled exceptions |
| AUTHENTICATION_ERROR | Authentication failed | 401 | Invalid credentials |
| SERVICE_UNAVAILABLE | Service unavailable | 503 | Server down, maintenance |
