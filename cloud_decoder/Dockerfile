# cloud_decoder/Dockerfile

# Use official Python image with CUDA support for GPU inference
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=4 \
    TOKENIZERS_PARALLELISM=false

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    wget \
    git \
    libgomp1 \
    liblz4-dev \
    libmsgpack-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for Python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support (separate for better caching)
RUN pip install --no-cache-dir \
    torch==2.1.0+cu118 \
    torchvision==0.16.0+cu118 \
    torchaudio==2.1.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Copy decoder code
COPY optimized_decoder.py .
COPY __init__.py .

# Copy vocabulary manager (needs to be available)
COPY ../vocabulary/vocabulary_manager.py ./vocabulary/
COPY ../vocabulary/__init__.py ./vocabulary/

# Create directories for models and logs
RUN mkdir -p /app/models /app/logs /app/vocabs

# Download models if needed (optional - can mount instead)
# RUN python -c "from optimized_decoder import download_models; download_models()"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "optimized_decoder:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--log-level", "info"]

# Alternative for production with gunicorn
# CMD ["gunicorn", "optimized_decoder:app", \
#      "-w", "1", \
#      "-k", "uvicorn.workers.UvicornWorker", \
#      "--bind", "0.0.0.0:8000", \
#      "--timeout", "120", \
#      "--keep-alive", "5", \
#      "--max-requests", "1000", \
#      "--max-requests-jitter", "50"]