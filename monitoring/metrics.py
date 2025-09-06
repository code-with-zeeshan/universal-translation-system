# monitoring/metrics.py
"""
Central metrics definitions for the Universal Translation System.
This file defines all metrics used across the system to ensure consistency.
"""

from prometheus_client import Counter, Histogram, Gauge, Summary
import logging

logger = logging.getLogger(__name__)

# Translation metrics
TRANSLATION_REQUESTS = Counter(
    'translation_requests_total',
    'Total number of translation requests',
    ['source_lang', 'target_lang', 'status']
)

TRANSLATION_LATENCY = Histogram(
    'translation_latency_seconds',
    'Translation request latency in seconds',
    ['source_lang', 'target_lang', 'component'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

TRANSLATION_TEXT_LENGTH = Histogram(
    'translation_text_length_chars',
    'Length of translated text in characters',
    ['source_lang', 'target_lang'],
    buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000]
)

# Vocabulary metrics
VOCABULARY_PACKS = Gauge(
    'vocabulary_packs_total',
    'Total number of vocabulary packs',
    ['status']  # loaded, available
)

VOCABULARY_TOKENS = Gauge(
    'vocabulary_tokens_total',
    'Total number of tokens in vocabulary',
    ['language', 'type']  # type: regular, subword, special
)

VOCABULARY_MEMORY_USAGE = Gauge(
    'vocabulary_memory_usage_bytes',
    'Memory usage of vocabulary packs in bytes',
    ['language']
)

VOCABULARY_LOAD_TIME = Histogram(
    'vocabulary_load_time_seconds',
    'Time to load vocabulary pack in seconds',
    ['language'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

# Encoder metrics
ENCODER_PROCESSING_TIME = Histogram(
    'encoder_processing_time_seconds',
    'Time to encode text in seconds',
    ['source_lang', 'target_lang'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
)

ENCODER_MEMORY_USAGE = Gauge(
    'encoder_memory_usage_bytes',
    'Memory usage of encoder in bytes'
)

ENCODER_ERRORS = Counter(
    'encoder_errors_total',
    'Total number of encoder errors',
    ['error_type', 'source_lang', 'target_lang']
)

# Decoder metrics
DECODER_PROCESSING_TIME = Histogram(
    'decoder_processing_time_seconds',
    'Time to decode embeddings in seconds',
    ['source_lang', 'target_lang'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

DECODER_MEMORY_USAGE = Gauge(
    'decoder_memory_usage_bytes',
    'Memory usage of decoder in bytes'
)

DECODER_ERRORS = Counter(
    'decoder_errors_total',
    'Total number of decoder errors',
    ['error_type', 'source_lang', 'target_lang']
)

DECODER_QUEUE_SIZE = Gauge(
    'decoder_queue_size',
    'Number of requests in decoder queue',
    ['decoder_id']
)

# Coordinator metrics
COORDINATOR_ACTIVE_CONNECTIONS = Gauge(
    'coordinator_active_connections',
    'Number of active connections to coordinator'
)

COORDINATOR_ROUTING_TIME = Histogram(
    'coordinator_routing_time_seconds',
    'Time to route request to decoder in seconds',
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
)

COORDINATOR_DECODER_HEALTH = Gauge(
    'coordinator_decoder_health',
    'Health status of decoder (1=healthy, 0=unhealthy)',
    ['decoder_id']
)

# System metrics
SYSTEM_CPU_USAGE = Gauge(
    'system_cpu_usage_percent',
    'CPU usage percentage'
)

# JWKS metrics
JWKS_RELOADS_SUCCESS = Counter(
    'jwks_reloads_success_total',
    'Total number of successful JWKS reloads',
    ['component']  # coordinator or decoder
)
JWKS_RELOADS_FAILURE = Counter(
    'jwks_reloads_failure_total',
    'Total number of failed JWKS reloads',
    ['component']
)
JWKS_KEYS = Gauge(
    'jwks_keys_total',
    'Current number of JWKS keys',
    ['component']
)

SYSTEM_MEMORY_USAGE = Gauge(
    'system_memory_usage_bytes',
    'Memory usage in bytes'
)

SYSTEM_GPU_MEMORY_USAGE = Gauge(
    'system_gpu_memory_usage_bytes',
    'GPU memory usage in bytes',
    ['gpu_id']
)

SYSTEM_GPU_UTILIZATION = Gauge(
    'system_gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id']
)

# API metrics
API_REQUESTS = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['endpoint', 'method', 'status_code']
)

API_LATENCY = Histogram(
    'api_latency_seconds',
    'API request latency in seconds',
    ['endpoint', 'method'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

# SDK metrics
SDK_REQUESTS = Counter(
    'sdk_requests_total',
    'Total number of SDK requests',
    ['sdk_type', 'method', 'status']
)

SDK_LATENCY = Histogram(
    'sdk_latency_seconds',
    'SDK request latency in seconds',
    ['sdk_type', 'method'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

# Helper functions for common metric operations

def track_translation_request(source_lang, target_lang, status, latency=None):
    """
    Track a translation request with metrics.
    
    Args:
        source_lang: Source language code
        target_lang: Target language code
        status: Status of the request ('success', 'error', etc.)
        latency: Latency of the request in seconds (optional)
    """
    TRANSLATION_REQUESTS.labels(
        source_lang=source_lang,
        target_lang=target_lang,
        status=status
    ).inc()
    
    if latency is not None and status == 'success':
        TRANSLATION_LATENCY.labels(
            source_lang=source_lang,
            target_lang=target_lang,
            component='total'
        ).observe(latency)

def track_encoder_processing(source_lang, target_lang, duration):
    """
    Track encoder processing time.
    
    Args:
        source_lang: Source language code
        target_lang: Target language code
        duration: Processing time in seconds
    """
    ENCODER_PROCESSING_TIME.labels(
        source_lang=source_lang,
        target_lang=target_lang
    ).observe(duration)
    
    # Also track as part of the overall latency
    TRANSLATION_LATENCY.labels(
        source_lang=source_lang,
        target_lang=target_lang,
        component='encoder'
    ).observe(duration)

def track_decoder_processing(source_lang, target_lang, duration):
    """
    Track decoder processing time.
    
    Args:
        source_lang: Source language code
        target_lang: Target language code
        duration: Processing time in seconds
    """
    DECODER_PROCESSING_TIME.labels(
        source_lang=source_lang,
        target_lang=target_lang
    ).observe(duration)
    
    # Also track as part of the overall latency
    TRANSLATION_LATENCY.labels(
        source_lang=source_lang,
        target_lang=target_lang,
        component='decoder'
    ).observe(duration)

def track_encoder_error(error_type, source_lang, target_lang):
    """
    Track an encoder error.
    
    Args:
        error_type: Type of error
        source_lang: Source language code
        target_lang: Target language code
    """
    ENCODER_ERRORS.labels(
        error_type=error_type,
        source_lang=source_lang,
        target_lang=target_lang
    ).inc()
    
    # Also track as a failed translation request
    TRANSLATION_REQUESTS.labels(
        source_lang=source_lang,
        target_lang=target_lang,
        status='error'
    ).inc()

def track_decoder_error(error_type, source_lang, target_lang):
    """
    Track a decoder error.
    
    Args:
        error_type: Type of error
        source_lang: Source language code
        target_lang: Target language code
    """
    DECODER_ERRORS.labels(
        error_type=error_type,
        source_lang=source_lang,
        target_lang=target_lang
    ).inc()
    
    # Also track as a failed translation request
    TRANSLATION_REQUESTS.labels(
        source_lang=source_lang,
        target_lang=target_lang,
        status='error'
    ).inc()

def update_vocabulary_metrics(language, tokens_count, memory_usage, load_time=None):
    """
    Update vocabulary metrics.
    
    Args:
        language: Language code
        tokens_count: Dictionary with token counts by type
        memory_usage: Memory usage in bytes
        load_time: Time to load vocabulary in seconds (optional)
    """
    for token_type, count in tokens_count.items():
        VOCABULARY_TOKENS.labels(
            language=language,
            type=token_type
        ).set(count)
    
    VOCABULARY_MEMORY_USAGE.labels(
        language=language
    ).set(memory_usage)
    
    if load_time is not None:
        VOCABULARY_LOAD_TIME.labels(
            language=language
        ).observe(load_time)

def update_system_metrics(cpu_usage, memory_usage, gpu_metrics=None):
    """
    Update system metrics.
    
    Args:
        cpu_usage: CPU usage percentage
        memory_usage: Memory usage in bytes
        gpu_metrics: Dictionary of GPU metrics by GPU ID (optional)
    """
    SYSTEM_CPU_USAGE.set(cpu_usage)
    SYSTEM_MEMORY_USAGE.set(memory_usage)
    
    if gpu_metrics:
        for gpu_id, metrics in gpu_metrics.items():
            if 'memory_usage' in metrics:
                SYSTEM_GPU_MEMORY_USAGE.labels(
                    gpu_id=gpu_id
                ).set(metrics['memory_usage'])
            
            if 'utilization' in metrics:
                SYSTEM_GPU_UTILIZATION.labels(
                    gpu_id=gpu_id
                ).set(metrics['utilization'])

def track_api_request(endpoint, method, status_code, latency):
    """
    Track an API request.
    
    Args:
        endpoint: API endpoint
        method: HTTP method
        status_code: HTTP status code
        latency: Request latency in seconds
    """
    API_REQUESTS.labels(
        endpoint=endpoint,
        method=method,
        status_code=status_code
    ).inc()
    
    API_LATENCY.labels(
        endpoint=endpoint,
        method=method
    ).observe(latency)

def track_sdk_request(sdk_type, method, status, latency=None):
    """
    Track an SDK request.
    
    Args:
        sdk_type: Type of SDK ('android', 'ios', 'web', etc.)
        method: Method name
        status: Status of the request ('success', 'error', etc.)
        latency: Request latency in seconds (optional)
    """
    SDK_REQUESTS.labels(
        sdk_type=sdk_type,
        method=method,
        status=status
    ).inc()
    
    if latency is not None:
        SDK_LATENCY.labels(
            sdk_type=sdk_type,
            method=method
        ).observe(latency)