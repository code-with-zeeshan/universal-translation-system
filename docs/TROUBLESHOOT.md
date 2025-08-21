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
# Install all dependencies
pip install -r requirements.txt

# Check for missing dependencies
python scripts/check_dependencies.py
```

### CUDA Issues
**Symptoms**: "CUDA not available" warnings, CPU-only operation
**Solution**:
1. Verify CUDA installation: `nvidia-smi`
2. Check PyTorch CUDA compatibility: `python -c "import torch; print(torch.cuda.is_available())"`
3. Reinstall PyTorch with CUDA support: `pip install torch==2.0.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html`

## SDK Integration Issues

### Android SDK Issues
**Symptoms**: JNI errors, crashes on initialization
**Solutions**:
1. Check NDK version compatibility
2. Verify that the native library is properly included
3. Add proper error handling:
```kotlin
try {
    val translator = TranslationClient(context)
    // Use translator
} catch (e: Exception) {
    Log.e("Translation", "Error initializing translator", e)
    // Show user-friendly error message
}
```

### iOS SDK Issues
**Symptoms**: Missing symbols, crashes on initialization
**Solutions**:
1. Check that the framework is properly linked
2. Verify that the native library is included in the bundle
3. Add proper error handling:
```swift
do {
    let translator = try TranslationClient()
    // Use translator
} catch {
    print("Error initializing translator: \(error)")
    // Show user-friendly error message
}
```

### Web SDK Issues
**Symptoms**: CORS errors, WebAssembly loading failures
**Solutions**:
1. Ensure CORS is properly configured on the server
2. Check browser compatibility for WebAssembly
3. Add fallback to API-based encoding:
```javascript
const translator = new TranslationClient({
    useWasm: true,  // Will fall back to API if WASM fails
    decoderUrl: "https://api.example.com/decode"
});
```

## Encoder Issues

### Encoding Failures
**Symptoms**: ENCODING_FAILED errors, empty embeddings
**Solutions**:
1. Check vocabulary availability:
```python
# Python
if not encoder.has_vocabulary(source_lang):
    await encoder.load_vocabulary(source_lang)
```

2. Verify input text format and encoding
3. Check for memory issues:
```python
# Monitor memory usage
import psutil
print(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
```

### Vocabulary Loading Failures
**Symptoms**: VOCABULARY_NOT_LOADED errors
**Solutions**:
1. Check vocabulary file existence and permissions
2. Verify vocabulary format compatibility
3. Try downloading vocabulary again:
```python
# Python
await encoder.download_vocabulary(source_lang)
```

## Decoder Issues

### Decoder Connection Issues
**Symptoms**: NETWORK_ERROR, connection timeouts
**Solutions**:
1. Check decoder endpoint availability:
```bash
curl -v https://your-decoder-endpoint.com/health
```

2. Verify network connectivity and firewall settings
3. Check for rate limiting or IP blocking
4. Implement retry logic:
```python
# Python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def translate_with_retry(client, text, source_lang, target_lang):
    return await client.translate(text, source_lang, target_lang)
```

### Decoder Performance Issues
**Symptoms**: Slow translations, timeouts
**Solutions**:
1. Check decoder load and scaling
2. Verify GPU availability and utilization
3. Implement request batching:
```python
# Python
async def translate_batch(client, texts, source_lang, target_lang):
    return await client.translate_batch(texts, source_lang, target_lang)
```

## Vocabulary Issues

### Missing Vocabulary
**Symptoms**: VOCABULARY_NOT_LOADED errors
**Solutions**:
1. Check vocabulary directory structure
2. Verify language code correctness
3. Download vocabulary manually:
```python
# Python
from vocabulary.downloader import VocabularyDownloader
downloader = VocabularyDownloader()
await downloader.download_vocabulary(language_code)
```

### Vocabulary Format Issues
**Symptoms**: Parsing errors, encoding failures
**Solutions**:
1. Verify vocabulary file format
2. Check for corruption in vocabulary files
3. Regenerate vocabulary:
```bash
python vocabulary/create_vocabulary.py --lang <language_code> --data <path_to_data>
```

## Performance Issues

### Slow Translation
**Symptoms**: High latency, timeouts
**Solutions**:
1. Check network latency between client and server
2. Verify decoder GPU utilization
3. Implement batching for multiple translations
4. Consider using a closer decoder node
5. Check for memory leaks:
```python
# Monitor memory usage over time
import psutil
import time

start_mem = psutil.Process().memory_info().rss
for i in range(100):
    # Perform translation
    time.sleep(1)
    current_mem = psutil.Process().memory_info().rss
    print(f"Memory usage: {current_mem / 1024 / 1024:.2f} MB, Delta: {(current_mem - start_mem) / 1024 / 1024:.2f} MB")
```

### High Memory Usage
**Symptoms**: Out of memory errors, crashes
**Solutions**:
1. Reduce batch size
2. Implement memory-efficient vocabulary loading
3. Use streaming for large translations
4. Check for memory leaks in vocabulary management

## Deployment Issues

### Docker Deployment Issues
**Symptoms**: Container fails to start, crashes
**Solutions**:
1. Check Docker logs:
```bash
docker logs <container_id>
```
2. Verify GPU availability in container:
```bash
docker exec <container_id> nvidia-smi
```
3. Check volume mounts and permissions
4. Verify environment variables

### Kubernetes Deployment Issues
**Symptoms**: Pods fail to start, crash loops
**Solutions**:
1. Check pod logs:
```bash
kubectl logs <pod_name>
```
2. Verify GPU availability:
```bash
kubectl exec <pod_name> -- nvidia-smi
```
3. Check resource limits and requests
4. Verify persistent volume claims
5. Check for node affinity issues

## Error Recovery Strategies

### Circuit Breaker Pattern

The system implements a circuit breaker pattern to prevent cascading failures when decoder nodes become unavailable. This pattern has three states:

1. **CLOSED**: Normal operation, requests are passed through
2. **OPEN**: Service is failing, requests are rejected immediately
3. **HALF_OPEN**: Testing if the service is working again

When a decoder node fails repeatedly, the circuit breaker opens and stops sending requests to that node. After a recovery timeout, it allows a test request through. If successful, the circuit closes again; if not, it remains open.

**Implementation**:
- The `CircuitBreaker` class in `coordinator/circuit_breaker.py` handles this logic
- The coordinator uses circuit breakers for each decoder node
- Circuit breaker status is exposed via the coordinator's `/status` endpoint

**Configuration**:
```yaml
# In coordinator config
circuit_breaker:
  failure_threshold: 5  # Number of failures before opening
  recovery_timeout: 30  # Seconds to wait before testing again
  timeout: 10  # Request timeout in seconds
```

### Retry Logic

For transient failures, the system implements retry logic with exponential backoff:

```python
# Python example
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10)
)
async def translate_with_retry(client, text, source_lang, target_lang):
    return await client.translate(text, source_lang, target_lang)
```

**SDK Implementation**:
- All SDKs implement retry logic for network errors
- Configurable retry count and backoff parameters
- Automatic fallback to alternative decoder nodes

### Fallback Strategies

The system implements several fallback strategies:

1. **Decoder Node Fallback**: If a decoder node fails, the coordinator routes to another node
2. **Encoding Fallback**: If on-device encoding fails, SDKs can fall back to API-based encoding
3. **Language Fallback**: If a requested language pair isn't available, the system can use a pivot language

**Configuration**:
```yaml
# In client config
fallback:
  enable_api_encoding: true  # Fall back to API-based encoding
  enable_pivot_translation: true  # Enable pivot translation
  pivot_language: "en"  # Use English as pivot language
```

## Common Error Codes

The system uses standardized error codes across all components:

| Error Code | Description | HTTP Status | Common Causes |
|------------|-------------|-------------|--------------|
| `NETWORK_ERROR` | Network connectivity issues | 503 | Connection failures, timeouts |
| `MODEL_NOT_FOUND` | Model file not found | 404 | Missing model files, incorrect paths |
| `VOCABULARY_NOT_LOADED` | Vocabulary not loaded | 404 | Missing vocabulary files, download failures |
| `ENCODING_FAILED` | Text encoding failed | 500 | Invalid input, memory issues |
| `DECODING_FAILED` | Embedding decoding failed | 500 | Invalid embeddings, model errors |
| `INVALID_LANGUAGE` | Unsupported language | 400 | Typos in language codes, unsupported languages |
| `RATE_LIMITED` | Too many requests | 429 | Exceeding API limits |
| `RESOURCE_EXHAUSTED` | Out of resources | 503 | Out of memory, GPU memory exhausted |
| `TIMEOUT` | Request timed out | 504 | Slow network, overloaded servers |
| `INVALID_INPUT` | Invalid input parameters | 400 | Malformed requests, invalid parameters |
| `INTERNAL_ERROR` | Unexpected internal error | 500 | Bugs, unhandled exceptions |
| `AUTHENTICATION_ERROR` | Authentication failed | 401 | Invalid credentials |
| `AUTHORIZATION_ERROR` | Authorization failed | 403 | Insufficient permissions |
| `SERVICE_UNAVAILABLE` | Service unavailable | 503 | Server down, maintenance |

**Error Handling Example**:
```python
# Python
from utils.error_codes import TranslationError, TranslationErrorCode

try:
    result = await client.translate(text, source_lang, target_lang)
except TranslationError as e:
    if e.code == TranslationErrorCode.VOCABULARY_NOT_LOADED:
        # Try to download vocabulary
        await client.download_vocabulary(source_lang)
        # Retry
        result = await client.translate(text, source_lang, target_lang)
    elif e.code == TranslationErrorCode.NETWORK_ERROR:
        # Retry with exponential backoff
        # ...
    else:
        # Handle other errors
        print(f"Translation error: {e.code.value} - {e.message}")
```