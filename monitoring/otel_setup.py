"""
Sentinel AI — OpenTelemetry Setup

Initializes the OpenTelemetry tracing and metrics pipeline.
Supports exporting to OTLP (Jaeger/Tempo) and Prometheus.

Usage:
    from monitoring.otel_setup import setup_otel, get_tracer, get_meter

    setup_otel()  # call once at application startup
    tracer = get_tracer("sentinel.pipeline")
    meter  = get_meter("sentinel.metrics")

    with tracer.start_as_current_span("vision_inference") as span:
        span.set_attribute("model.name", "llava:7b")
        ...
"""
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level tracer / meter (initialized by setup_otel)
_tracer_provider = None
_meter_provider = None


def setup_otel(
    service_name: str = "sentinel-ai",
    service_version: str = "0.2.0",
    otlp_endpoint: str = "http://localhost:4317",
    enabled: bool = True,
) -> None:
    """
    Initialize OpenTelemetry SDK for tracing and metrics.

    Sets up:
    - TracerProvider with OTLP gRPC exporter (→ Jaeger/Tempo)
    - MeterProvider with OTLP metrics exporter (→ Prometheus via collector)
    - Batch span processor for efficient exporting

    Args:
        service_name: Logical service name (appears in Jaeger/Grafana).
        service_version: Semantic version of the service.
        otlp_endpoint: OTLP collector endpoint.
        enabled: If False, uses a no-op provider (for testing).
    """
    global _tracer_provider, _meter_provider

    if not enabled:
        logger.info("OpenTelemetry disabled — using no-op providers.")
        _setup_noop()
        return

    try:
        from opentelemetry import trace, metrics
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

        # Resource attributes (appear in every trace/metric)
        resource = Resource(attributes={
            SERVICE_NAME: service_name,
            SERVICE_VERSION: service_version,
        })

        # ── Tracing ──────────────────────────────────────
        span_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        span_processor = BatchSpanProcessor(span_exporter)

        _tracer_provider = TracerProvider(resource=resource)
        _tracer_provider.add_span_processor(span_processor)
        trace.set_tracer_provider(_tracer_provider)

        # ── Metrics ──────────────────────────────────────
        metric_exporter = OTLPMetricExporter(endpoint=otlp_endpoint, insecure=True)
        metric_reader = PeriodicExportingMetricReader(
            metric_exporter, export_interval_millis=15_000
        )
        _meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(_meter_provider)

        logger.info(
            f"✅ OpenTelemetry initialized — service={service_name}, "
            f"endpoint={otlp_endpoint}"
        )

    except ImportError as e:
        logger.warning(f"OpenTelemetry packages not installed: {e}. Using no-op.")
        _setup_noop()


def _setup_noop() -> None:
    """Set up no-op OTEL providers (for testing or when disabled)."""
    global _tracer_provider, _meter_provider
    try:
        from opentelemetry import trace, metrics
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.metrics import MeterProvider

        _tracer_provider = TracerProvider()
        _meter_provider = MeterProvider()
        trace.set_tracer_provider(_tracer_provider)
        metrics.set_meter_provider(_meter_provider)
    except ImportError:
        pass  # OTEL not available — completely skip


def get_tracer(name: str = "sentinel"):
    """
    Get an OpenTelemetry tracer.

    Args:
        name: Tracer scope name (usually the module or component name).

    Returns:
        OpenTelemetry Tracer instance.
    """
    try:
        from opentelemetry import trace
        return trace.get_tracer(name)
    except ImportError:
        return _NoopTracer()


def get_meter(name: str = "sentinel"):
    """
    Get an OpenTelemetry meter for custom metrics.

    Args:
        name: Meter scope name.

    Returns:
        OpenTelemetry Meter instance.
    """
    try:
        from opentelemetry import metrics
        return metrics.get_meter(name)
    except ImportError:
        return _NoopMeter()


def shutdown_otel() -> None:
    """Flush and shutdown all OTEL providers (call on app shutdown)."""
    global _tracer_provider, _meter_provider
    try:
        if _tracer_provider:
            _tracer_provider.shutdown()
        if _meter_provider:
            _meter_provider.shutdown()
        logger.info("OpenTelemetry shut down cleanly.")
    except Exception as e:
        logger.warning(f"OTEL shutdown error: {e}")


# ─────────────────────────────────────────
# No-op fallbacks for test environments
# ─────────────────────────────────────────

class _NoopSpan:
    """A no-op span context manager for when OTEL is unavailable."""
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def set_attribute(self, key, value): pass
    def set_status(self, *args): pass
    def record_exception(self, exc): pass
    def add_event(self, name, attributes=None): pass


class _NoopTracer:
    """A no-op tracer for when OTEL is unavailable."""
    def start_as_current_span(self, name, **kwargs):
        return _NoopSpan()

    def start_span(self, name, **kwargs):
        return _NoopSpan()


class _NoopMeter:
    """A no-op meter for when OTEL is unavailable."""
    def create_histogram(self, *args, **kwargs): return _NoopInstrument()
    def create_counter(self, *args, **kwargs): return _NoopInstrument()
    def create_gauge(self, *args, **kwargs): return _NoopInstrument()
    def create_up_down_counter(self, *args, **kwargs): return _NoopInstrument()


class _NoopInstrument:
    """A no-op metric instrument."""
    def record(self, *args, **kwargs): pass
    def add(self, *args, **kwargs): pass
    def observe(self, *args, **kwargs): pass
