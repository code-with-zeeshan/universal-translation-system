# docker/decoder.Dockerfile (standardized)
# Base: CUDA 11.8 runtime on Ubuntu 22.04, production-oriented
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# System deps and Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    curl \
    wget \
    git \
    liblz4-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Ensure python command is available
RUN ln -s /usr/bin/python3 /usr/bin/python || true \
 && ln -s /usr/bin/pip3 /usr/bin/pip || true

WORKDIR /app

# Copy canonical requirements
COPY requirements/base.txt /app/requirements/base.txt
COPY requirements/serve.txt /app/requirements/serve.txt
COPY requirements/decoder.txt /app/requirements/decoder.txt

# Upgrade pip and install Python dependencies from canonical sets
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements/base.txt -r /app/requirements/serve.txt -r /app/requirements/decoder.txt

# Install PyTorch with CUDA 11.8 explicitly (pinned for stability)
RUN pip install --no-cache-dir \
    torch==2.1.0+cu118 \
    torchvision==0.16.0+cu118 \
    torchaudio==2.1.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Copy decoder code and required shared modules
COPY cloud_decoder /app
COPY utils /app/utils
COPY vocabulary /app/vocabulary
COPY monitoring /app/monitoring

# Runtime env tuning
ENV OMP_NUM_THREADS=4 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    CUDA_LAUNCH_BLOCKING=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8001 \
    API_WORKERS=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/app

# Create non-root user and fix perms
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app

EXPOSE 8001

# Healthcheck (respects API_PORT)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD ["bash", "-lc", "curl -fsS http://localhost:${API_PORT:-8001}/health || exit 1"]

USER appuser

# Production runner: Litserve (fast, aligned with repo design)
CMD ["bash", "-lc", "litserve serve optimized_decoder:app --host 0.0.0.0 --port ${API_PORT:-8001}"]