# docker/coordinator.Dockerfile — Coordinator service
# Multi-stage build for minimal production image

# Stage 1: Python dependencies
FROM python:3.14-slim AS builder

WORKDIR /app

COPY requirements/base.txt requirements/base.txt
COPY requirements/serve.txt requirements/serve.txt
COPY requirements/coordinator.txt requirements/coordinator.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        -r requirements/base.txt \
        -r requirements/serve.txt \
        -r requirements/coordinator.txt

# Stage 2: Runtime
FROM python:3.14-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN adduser --disabled-password --gecos '' appuser

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY coordinator/ coordinator/
COPY utils/ utils/
COPY config/ config/
COPY version-config.json .

RUN mkdir -p /app/logs /app/config && chown -R appuser:appuser /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    COORDINATOR_HOST=0.0.0.0 \
    COORDINATOR_PORT=5100 \
    COORDINATOR_WORKERS=1

EXPOSE 5100

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:${COORDINATOR_PORT:-5100}/health || exit 1

USER appuser

ENTRYPOINT ["uvicorn", "coordinator.advanced_coordinator:app", "--host", "0.0.0.0", "--port", "5100"]
# Override port via COORDINATOR_PORT env var (default 5100)
