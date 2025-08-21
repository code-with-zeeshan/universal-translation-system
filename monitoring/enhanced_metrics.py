"""
Enhanced metrics for the Universal Translation System.
Provides detailed metrics for all components using Prometheus.
"""

from prometheus_client import Counter, Gauge, Histogram, Summary
import time
from typing import Dict, Any, Optional, Callable
import functools
import asyncio

# Translation metrics
TRANSLATION_REQUESTS = Counter(
    'translation_requests_total', 
    'Total number of translation requests',
    ['source_lang', 'target_lang', 'sdk_type', 'status']
)

TRANSLATION_LATENCY = Histogram(
    'translation_latency_seconds',
    'Translation request latency in seconds',
    ['source_lang', 'target_lang', 'sdk_type', 'component'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
)

# Encoder metrics
ENCODING_SIZE = Histogram(
    'encoding_size_bytes',
    'Size of encoded data in bytes',
    ['source_lang', 'target_lang'],
    buckets=(100, 500, 1000, 2000, 5000, 10000, 20000)
)

# Decoder metrics
DECODER_QUEUE_SIZE = Gauge(
    'decoder_queue_size',
    'Number of requests in decoder queue',
    ['decoder_id']
)

DECODER_GPU_UTILIZATION = Gauge(
    'decoder_gpu_utilization_percent',
    'GPU utilization percentage',
    ['decoder_id', 'gpu_id']
)

DECODER_GPU_MEMORY = Gauge(
    'decoder_gpu_memory_used_bytes',
    'GPU memory used in bytes',
    ['decoder_id', 'gpu_id']
)

# Vocabulary metrics
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
    ['decision_type']  # 'least_loaded', 'nearest', 'fallback'
)

# Circuit breaker metrics
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

# System metrics
SYSTEM_MEMORY_USAGE = Gauge(
    'system_memory_usage_bytes',
    'System memory usage in bytes'
)

SYSTEM_CPU_USAGE = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage'
)

# Error metrics
ERROR_COUNT = Counter(
    'error_count_total',
    'Total number of errors',
    ['component', 'error_code']
)

# Helper functions
def track_translation_request(source_lang, target_lang, sdk_type, status="success"):
    """Track a translation request"""
    TRANSLATION_REQUESTS.labels(
        source_lang=source_lang,
        target_lang=target_lang,
        sdk_type=sdk_type,
        status=status
    ).inc()

def track_translation_latency(source_lang, target_lang, sdk_type, component, seconds):
    """Track translation latency"""
    TRANSLATION_LATENCY.labels(
        source_lang=source_lang,
        target_lang=target_lang,
        sdk_type=sdk_type,
        component=component
    ).observe(seconds)

def track_encoding_size(source_lang, target_lang, size_bytes):
    """Track encoding size"""
    ENCODING_SIZE.labels(
        source_lang=source_lang,
        target_lang=target_lang
    ).observe(size_bytes)

def track_error(component, error_code):
    """Track an error"""
    ERROR_COUNT.labels(
        component=component,
        error_code=error_code
    ).inc()

def track_vocabulary_cache_hit(language):
    """Track a vocabulary cache hit"""
    VOCABULARY_CACHE_HITS.labels(language=language).inc()

def track_vocabulary_cache_miss(language):
    """Track a vocabulary cache miss"""
    VOCABULARY_CACHE_MISSES.labels(language=language).inc()

def set_vocabulary_cache_size(language, size_bytes):
    """Set vocabulary cache size"""
    VOCABULARY_CACHE_SIZE.labels(language=language).set(size_bytes)

def set_decoder_queue_size(decoder_id, size):
    """Set decoder queue size"""
    DECODER_QUEUE_SIZE.labels(decoder_id=decoder_id).set(size)

def set_decoder_gpu_utilization(decoder_id, gpu_id, utilization):
    """Set decoder GPU utilization"""
    DECODER_GPU_UTILIZATION.labels(
        decoder_id=decoder_id,
        gpu_id=gpu_id
    ).set(utilization)

def set_decoder_gpu_memory(decoder_id, gpu_id, memory_bytes):
    """Set decoder GPU memory usage"""
    DECODER_GPU_MEMORY.labels(
        decoder_id=decoder_id,
        gpu_id=gpu_id
    ).set(memory_bytes)

def set_coordinator_active_decoders(count):
    """Set number of active decoders"""
    COORDINATOR_ACTIVE_DECODERS.set(count)

def track_coordinator_routing_decision(decision_type):
    """Track a coordinator routing decision"""
    COORDINATOR_ROUTING_DECISIONS.labels(decision_type=decision_type).inc()

def set_circuit_breaker_state(name, state):
    """Set circuit breaker state"""
    # Convert state to numeric value
    state_value = {
        "CLOSED": 0,
        "HALF_OPEN": 1,
        "OPEN": 2
    }.get(state, 0)
    
    CIRCUIT_BREAKER_STATE.labels(name=name).set(state_value)

def track_circuit_breaker_failure(name, error_type):
    """Track a circuit breaker failure"""
    CIRCUIT_BREAKER_FAILURES.labels(
        name=name,
        error_type=error_type
    ).inc()

def track_circuit_breaker_success(name):
    """Track a circuit breaker success"""
    CIRCUIT_BREAKER_SUCCESSES.labels(name=name).inc()

def set_system_memory_usage(memory_bytes):
    """Set system memory usage"""
    SYSTEM_MEMORY_USAGE.set(memory_bytes)

def set_system_cpu_usage(cpu_percent):
    """Set system CPU usage"""
    SYSTEM_CPU_USAGE.set(cpu_percent)

class LatencyTracker:
    """Context manager for tracking latency"""
    
    def __init__(self, source_lang, target_lang, sdk_type, component):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.sdk_type = sdk_type
        self.component = component
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        latency = time.time() - self.start_time
        track_translation_latency(
            self.source_lang,
            self.target_lang,
            self.sdk_type,
            self.component,
            latency
        )
        
        # Track errors if any
        if exc_type is not None:
            track_error(self.component, exc_type.__name__)

class AsyncLatencyTracker:
    """Async context manager for tracking latency"""
    
    def __init__(self, source_lang, target_lang, sdk_type, component):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.sdk_type = sdk_type
        self.component = component
        self.start_time = None
        
    async def __aenter__(self):
        self.start_time = time.time()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        latency = time.time() - self.start_time
        track_translation_latency(
            self.source_lang,
            self.target_lang,
            self.sdk_type,
            self.component,
            latency
        )
        
        # Track errors if any
        if exc_type is not None:
            track_error(self.component, exc_type.__name__)

def track_latency(source_lang, target_lang, sdk_type, component):
    """
    Decorator for tracking function latency
    
    Args:
        source_lang: Source language
        target_lang: Target language
        sdk_type: SDK type
        component: Component name
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with LatencyTracker(source_lang, target_lang, sdk_type, component):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def track_async_latency(source_lang, target_lang, sdk_type, component):
    """
    Decorator for tracking async function latency
    
    Args:
        source_lang: Source language
        target_lang: Target language
        sdk_type: SDK type
        component: Component name
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            async with AsyncLatencyTracker(source_lang, target_lang, sdk_type, component):
                return await func(*args, **kwargs)
        return wrapper
    return decorator

# System metrics collection
def start_system_metrics_collection(interval=15):
    """
    Start collecting system metrics at regular intervals
    
    Args:
        interval: Collection interval in seconds
    """
    import psutil
    import threading
    
    def collect_metrics():
        while True:
            try:
                # Memory usage
                memory = psutil.virtual_memory()
                set_system_memory_usage(memory.used)
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                set_system_cpu_usage(cpu_percent)
                
                # GPU metrics (if available)
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    for i, gpu in enumerate(gpus):
                        set_decoder_gpu_utilization("local", str(i), gpu.load * 100)
                        set_decoder_gpu_memory("local", str(i), gpu.memoryUsed * 1024 * 1024)
                except ImportError:
                    pass
                
                # Wait for next collection
                time.sleep(interval)
            except Exception as e:
                print(f"Error collecting system metrics: {e}")
                time.sleep(interval)
    
    # Start collection thread
    thread = threading.Thread(target=collect_metrics, daemon=True)
    thread.start()
    
    return thread

async def start_async_system_metrics_collection(interval=15):
    """
    Start collecting system metrics at regular intervals (async version)
    
    Args:
        interval: Collection interval in seconds
    """
    import psutil
    
    async def collect_metrics():
        while True:
            try:
                # Memory usage
                memory = psutil.virtual_memory()
                set_system_memory_usage(memory.used)
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                set_system_cpu_usage(cpu_percent)
                
                # GPU metrics (if available)
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    for i, gpu in enumerate(gpus):
                        set_decoder_gpu_utilization("local", str(i), gpu.load * 100)
                        set_decoder_gpu_memory("local", str(i), gpu.memoryUsed * 1024 * 1024)
                except ImportError:
                    pass
                
                # Wait for next collection
                await asyncio.sleep(interval)
            except Exception as e:
                print(f"Error collecting system metrics: {e}")
                await asyncio.sleep(interval)
    
    # Start collection task
    task = asyncio.create_task(collect_metrics())
    
    return task