# docker/decoder.Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY cloud_decoder/requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip3 install -r requirements.txt

# Install Triton for optimized inference
RUN pip3 install triton

# Copy decoder code
COPY cloud_decoder /app

# Optimize for production
ENV OMP_NUM_THREADS=4
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CUDA_LAUNCH_BLOCKING=0

CMD ["python3", "optimized_decoder.py"]
