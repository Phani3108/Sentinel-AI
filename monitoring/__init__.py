"""
Sentinel AI — Monitoring Package
Provides OTEL and Prometheus monitoring utilities.
"""
from .otel_setup import setup_otel, get_tracer, get_meter, shutdown_otel
from .prometheus_metrics import METRICS, SentinelMetrics

__all__ = [
    "setup_otel",
    "get_tracer",
    "get_meter",
    "shutdown_otel",
    "METRICS",
    "SentinelMetrics",
]
