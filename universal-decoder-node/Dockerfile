FROM python:3.14-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY udn/ /app/udn/
COPY setup.py pyproject.toml README.md /app/

RUN pip install --no-cache-dir -e ".[gpu]"

EXPOSE 8000

ENV VOCAB_DIR=/app/vocabulary/vocab
ENV MODEL_PATH=/app/models/production/decoder.pt

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["udn"]
CMD ["start", "--host", "0.0.0.0", "--port", "8000"]
