# Deployment Guide

## Encoder Deployment (Mobile)

### Android

1. Add encoder to your app:
```bash
app/src/main/assets/ 
├── models/ 
│ └── universal_encoder.onnx 
└── vocabularies/ 
    └── latin_v1.msgpack
```

2. Add dependency:
```gradle
dependencies {
    implementation 'com.universaltranslation:encoder-sdk:1.0.0'
}
```

### iOS

1. Add to Xcode project:
- Drag **UniversalEncoder.mlmodelc** to project
- Ensure "Copy items if needed" is checked

2. Install via CocoaPods:
```Podfile
pod 'UniversalTranslationSDK', '~> 1.0'
```
### Web
```Javascript
<script src="https://cdn.example.com/universal-translation-sdk.min.js"></script>
```

Or via npm:
```bash
npm install universal-translation-sdk
```
## Decoder Deployment (Server)

### Docker Deployment
```bash
# Build image
docker build -f cloud_decoder/Dockerfile -t universal-decoder:latest .

# Run with GPU support
docker run --gpus all -p 8000:8000 \
  -e MAX_BATCH_SIZE=64 \
  -e GPU_MEMORY_FRACTION=0.9 \
  universal-decoder:latest
```

### Kubernetes Deployment
```yaml
# kubernetes/decoder-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: universal-decoder
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: decoder
        image: universal-decoder:latest
        resources:
          limits:
            nvidia.com/gpu: 1
```

Apply:
```bash
kubectl apply -f kubernetes/
```

### Cloud Platforms
**AWS**
- Use EC2 with GPU (g4dn.xlarge minimum)
- Or use SageMaker for managed deployment

**Google Cloud**
- Use Compute Engine with T4 GPU
- Or use Vertex AI for managed deployment

**Azure**
- Use NC-series VMs
- Or use Azure ML for managed deployment

## Environment Variables

### Decoder Service
- **MAX_BATCH_SIZE**: Maximum batch size (default: 32)
- **GPU_MEMORY_FRACTION**: GPU memory to use (default: 0.9)
- **MODEL_PATH**: Path to decoder model
- **PORT**: Service port (default: 8000)

## Monitoring

### Health Check
```bash
curl http://decoder-service:8000/health
```
### Metrics
The decoder exposes Prometheus metrics at **/metrics**

## Scaling

### Horizontal Scaling
- Add more decoder replicas
- Use load balancer to distribute requests

### Vertical Scaling
- Use larger GPUs (V100, A100)
- Increase batch size

### Security
1. **API Authentication**: Implement API keys or JWT
2. **Rate Limiting**: Limit requests per client
3. **HTTPS**: Always use TLS in production
4. **Input Validation**: Limit text length, validate language codes