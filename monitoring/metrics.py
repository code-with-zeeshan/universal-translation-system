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


# ── Enhanced metrics (ported from enhanced_metrics.py) ──────────────

# Encoder metrics
ENCODING_SIZE = Histogram(
    'encoding_size_bytes',
    'Size of encoded data in bytes',
    ['source_lang', 'target_lang'],
    buckets=(100, 500, 1000, 2000, 5000, 10000, 20000)
)

# Decoder GPU metrics
DECODER_GPU_UTILIZATION = Gauge(
    'decoder_gpu_utilization_percent',
    'GPU utilization percentage per decoder',
    ['decoder_id', 'gpu_id']
)

DECODER_GPU_MEMORY = Gauge(
    'decoder_gpu_memory_used_bytes',
    'GPU memory used in bytes per decoder',
    ['decoder_id', 'gpu_id']
)

# Vocabulary cache metrics
VOCABULARY_CACHE_SIZE = Gauge(
    'vocabulary_cache_size_bytes',
    'Size of vocabulary cache in bytes',
    ['language']
)

VOCABULARY_CACHE_HITS = Counter(
    'vocabulary_cache_hits_total',
    'Total number of vocabulary cache hits',
    ['language']
)

VOCABULARY_CACHE_MISSES = Counter(
    'vocabulary_cache_misses_total',
    'Total number of vocabulary cache misses',
    ['language']
)

# Coordinator metrics
COORDINATOR_ACTIVE_DECODERS = Gauge(
    'coordinator_active_decoders',
    'Number of active decoders in the pool'
)

COORDINATOR_ROUTING_DECISIONS = Counter(
    'coordinator_routing_decisions_total',
    'Total number of routing decisions',
    ['decision_type']
)

# Circuit breaker metrics (for coordinator multi-decoder failover)
CIRCUIT_BREAKER_STATE = Gauge(
    'circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=half-open, 2=open)',
    ['name']
)

CIRCUIT_BREAKER_FAILURES = Counter(
    'circuit_breaker_failures_total',
    'Total number of circuit breaker failures',
    ['name', 'error_type']
)

CIRCUIT_BREAKER_SUCCESSES = Counter(
    'circuit_breaker_successes_total',
    'Total number of circuit breaker successes',
    ['name']
)

# Error metrics
ERROR_COUNT = Counter(
    'error_count_total',
    'Total number of errors',
    ['component', 'error_code']
)


def track_encoding_size(source_lang: str, target_lang: str, size_bytes: int):
    """Track the size of encoded data in bytes."""
    ENCODING_SIZE.labels(source_lang=source_lang, target_lang=target_lang).observe(size_bytes)


def track_error(component: str, error_code: str):
    """Track an error by component and error code."""
    ERROR_COUNT.labels(component=component, error_code=error_code).inc()


def track_vocabulary_cache_hit(language: str):
    """Increment vocabulary cache hit counter for a language."""
    VOCABULARY_CACHE_HITS.labels(language=language).inc()


def track_vocabulary_cache_miss(language: str):
    """Increment vocabulary cache miss counter for a language."""
    VOCABULARY_CACHE_MISSES.labels(language=language).inc()


def set_vocabulary_cache_size(language: str, size_bytes: int):
    """Set the vocabulary cache size for a language."""
    VOCABULARY_CACHE_SIZE.labels(language=language).set(size_bytes)


def set_decoder_gpu_utilization(decoder_id: str, gpu_id: str, utilization: float):
    """Set GPU utilization percentage for a decoder node."""
    DECODER_GPU_UTILIZATION.labels(decoder_id=decoder_id, gpu_id=gpu_id).set(utilization)


def set_decoder_gpu_memory(decoder_id: str, gpu_id: str, memory_bytes: int):
    """Set GPU memory usage for a decoder node."""
    DECODER_GPU_MEMORY.labels(decoder_id=decoder_id, gpu_id=gpu_id).set(memory_bytes)


def set_coordinator_active_decoders(count: int):
    """Set the number of active decoders in the coordinator pool."""
    COORDINATOR_ACTIVE_DECODERS.set(count)


def track_coordinator_routing_decision(decision_type: str):
    """Increment routing decision counter by type."""
    COORDINATOR_ROUTING_DECISIONS.labels(decision_type=decision_type).inc()


def set_circuit_breaker_state(name: str, state: str):
    """Set circuit breaker state (CLOSED=0, HALF_OPEN=1, OPEN=2)."""
    state_value = {"CLOSED": 0, "HALF_OPEN": 1, "OPEN": 2}.get(state, 0)
    CIRCUIT_BREAKER_STATE.labels(name=name).set(state_value)


def track_circuit_breaker_failure(name: str, error_type: str):
    """Increment circuit breaker failure counter."""
    CIRCUIT_BREAKER_FAILURES.labels(name=name, error_type=error_type).inc()


def track_circuit_breaker_success(name: str):
    """Increment circuit breaker success counter."""
    CIRCUIT_BREAKER_SUCCESSES.labels(name=name).inc()