# monitoring/metrics_collector.py
from prometheus_client import Counter, Histogram, Gauge
import logging

# Metrics
translation_requests = Counter(
    'translation_requests_total',
    'Total translation requests',
    ['source_lang', 'target_lang', 'status']
)

translation_latency = Histogram(
    'translation_latency_seconds',
    'Translation latency',
    ['source_lang', 'target_lang'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

gpu_utilization = Gauge(
    'gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id']
)

active_connections = Gauge(
    'active_connections',
    'Number of active connections'
)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('translation_system.log'),
        logging.StreamHandler()
    ]
)