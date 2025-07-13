# docker/encoder.Dockerfile
FROM ubuntu:20.04 AS builder

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    cmake \
    git \
    wget

# Build ONNX Runtime for mobile
RUN git clone --recursive https://github.com/Microsoft/onnxruntime
WORKDIR /onnxruntime
RUN ./build.sh --config Release --build_shared_lib --parallel \
    --use_nnapi --use_coreml --minimal_build

# Build encoder libraries
COPY encoder_core /encoder_core
WORKDIR /encoder_core
RUN mkdir build && cd build && cmake .. && make -j$(nproc)

# Final stage
FROM ubuntu:20.04
COPY --from=builder /encoder_core/build/libuniversal_encoder.so /usr/lib/
COPY models /models
COPY vocabularies /vocabularies

---

# docker/decoder.Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch and other requirements
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Install Triton for optimized inference
RUN pip3 install triton

# Copy decoder code
COPY cloud_decoder /app
WORKDIR /app

# Optimize for production
ENV OMP_NUM_THREADS=4
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CUDA_LAUNCH_BLOCKING=0

CMD ["python3", "optimized_decoder.py"]