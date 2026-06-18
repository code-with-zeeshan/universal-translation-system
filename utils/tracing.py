"""Tracing utility — OpenTelemetry with graceful fallback.

Provides a unified interface for distributed tracing that works whether
OpenTelemetry packages are installed or not. When OTEL is unavailable,
all tracer calls become no-ops.

Usage:
    from utils.tracing import get_tracer, maybe_instrument_app, setup_tracing

    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("my_operation"):
        ...
"""

import logging
import os
from contextlib import nullcontext
from types import TracebackType
from typing import Optional, Type

logger = logging.getLogger(__name__)

_OTEL_AVAILABLE: bool = False
_trace = None  # module: opentelemetry.trace (or None)


class _NoOpSpan:
    """Drop-in replacement for OpenTelemetry Span — all methods are no-ops."""

    def __init__(self) -> None:
        self._attributes: dict = {}

    def set_attribute(self, key: str, value: object) -> None:
        self._attributes[key] = value

    def end(self) -> None:
        pass

    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        pass


class _NoOpTracer:
    """Drop-in replacement for opentelemetry.trace.Tracer."""

    def start_as_current_span(self, name: str, **kwargs: object) -> nullcontext:
        return nullcontext()

    def start_span(self, name: str, **kwargs: object) -> _NoOpSpan:
        return _NoOpSpan()

    def get_current_span(self) -> _NoOpSpan:
        return _NoOpSpan()


_NO_OP_TRACER = _NoOpTracer()


def get_tracer(module_name: str) -> object:
    """Return a tracer (real or no-op) for *module_name*."""
    if _trace is not None:
        return _trace.get_tracer(module_name)
    return _NO_OP_TRACER


def setup_tracing(service_name: str) -> None:
    """Configure the global tracer provider for *service_name*.

    Safe to call even when OTEL is not installed (no-op).
    """
    if not _OTEL_AVAILABLE:
        logger.info("OpenTelemetry not available — tracing disabled for %s", service_name)
        return
    try:
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

        provider = TracerProvider(
            resource=Resource.create({SERVICE_NAME: service_name})
        )
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        _trace.set_tracer_provider(provider)
        logger.info("Tracing enabled for %s (console exporter)", service_name)
    except Exception as exc:
        logger.warning("Failed to set up tracing for %s: %s", service_name, exc)


def setup_otlp_tracing(service_name: str, endpoint: Optional[str] = None) -> None:
    """Configure OTLP exporter for remote trace collection.

    Falls back to console exporter if OTLP is unavailable.
    """
    if not _OTEL_AVAILABLE:
        logger.info("OpenTelemetry not available — tracing disabled for %s", service_name)
        return
    try:
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        otlp_endpoint = endpoint or os.environ.get(
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "http://localhost:4318/v1/traces",
        )
        provider = TracerProvider(
            resource=Resource.create({SERVICE_NAME: service_name})
        )
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        _trace.set_tracer_provider(provider)
        logger.info("Tracing enabled for %s (OTLP: %s)", service_name, otlp_endpoint)
    except Exception as exc:
        logger.warning("OTLP tracing unavailable for %s, fallback to console: %s", service_name, exc)
        setup_tracing(service_name)


def maybe_instrument_app(app: object) -> None:
    """Instrument a FastAPI app for automatic request tracing.

    Safe to call when OTEL instrumentation is not installed (no-op).
    """
    if not _OTEL_AVAILABLE:
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)  # type: ignore
        logger.debug("FastAPI app instrumented for tracing")
    except Exception as exc:
        logger.debug("FastAPI instrumentation skipped: %s", exc)


def is_tracing_enabled() -> bool:
    return _OTEL_AVAILABLE


def shutdown_tracing() -> None:
    """Flush and shut down the tracer provider.

    Safe to call when OTEL is not installed (no-op).
    """
    if not _OTEL_AVAILABLE:
        return
    try:
        provider = _trace.get_tracer_provider()
        shutdown = getattr(provider, "shutdown", None)
        if callable(shutdown):
            shutdown()
    except Exception:
        pass


# --- Module-level init ---
try:
    from opentelemetry import trace as _trace
    _OTEL_AVAILABLE = True
except ImportError:
    _trace = None
    _OTEL_AVAILABLE = False
