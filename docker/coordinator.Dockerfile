FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY coordinator/requirements.txt coordinator-requirements.txt

# Install Python dependencies and add non-root user
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r coordinator-requirements.txt \
    && useradd -ms /bin/bash appuser

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/config \
    && chown -R appuser /app

# Set environment variables
ENV PYTHONPATH=/app
ENV COORDINATOR_HOST=0.0.0.0
ENV COORDINATOR_PORT=5100
ENV COORDINATOR_WORKERS=1
ENV COORDINATOR_TITLE="Universal Translation Coordinator"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 5100

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:5100/health || exit 1

# Drop privileges
USER appuser

# Run the coordinator
CMD ["python", "-m", "coordinator.main"]