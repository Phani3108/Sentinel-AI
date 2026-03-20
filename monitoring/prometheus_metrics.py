"""
Sentinel AI — Prometheus Metrics

Defines all custom Prometheus metrics for the Sentinel AI pipeline.
Metrics are exposed at /metrics via the prometheus-client library.

Metric Categories:
  1. Inference latency (vision, LLM, total pipeline)
  2. Request counters (total, by model, by outcome)
  3. Memory and resource usage
  4. RAG retrieval performance
  5. Error rates and retries

Usage:
    from monitoring.prometheus_metrics import METRICS

    with METRICS.vision_latency.labels(model="llava:7b").time():
        result = vision_model.analyze(...)

    METRICS.requests_total.labels(model="llava:7b", status="success").inc()
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


def _try_prometheus():
    """Return the prometheus_client module, or None if not installed."""
    try:
        import prometheus_client as prom
        return prom
    except ImportError:
        logger.warning(
            "prometheus_client not installed. Metrics will be no-ops. "
            "Install with: pip install prometheus-client"
        )
        return None


prom = _try_prometheus()


# ─────────────────────────────────────────────────────────────────────
# Metric Definitions
# ─────────────────────────────────────────────────────────────────────

def _histogram(name: str, doc: str, labels=(), buckets=None):
    """Create a Histogram or return no-op if prometheus unavailable."""
    if prom is None:
        return _NoopMetric()
    kwargs = {"labelnames": list(labels)}
    if buckets:
        kwargs["buckets"] = buckets
    # Use default registry; guard against duplicate registration
    try:
        return prom.Histogram(name, doc, **kwargs)
    except ValueError:
        return prom.REGISTRY._names_to_collectors.get(name, _NoopMetric())


def _counter(name: str, doc: str, labels=()):
    """Create a Counter or return no-op if prometheus unavailable."""
    if prom is None:
        return _NoopMetric()
    try:
        return prom.Counter(name, doc, labelnames=list(labels))
    except ValueError:
        return prom.REGISTRY._names_to_collectors.get(name, _NoopMetric())


def _gauge(name: str, doc: str, labels=()):
    """Create a Gauge or return no-op if prometheus unavailable."""
    if prom is None:
        return _NoopMetric()
    try:
        return prom.Gauge(name, doc, labelnames=list(labels))
    except ValueError:
        return prom.REGISTRY._names_to_collectors.get(name, _NoopMetric())


# Latency buckets (seconds): 0.1s to 120s
LATENCY_BUCKETS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0]

# ─────────────────────────────────────────────────────────────────────
# Centralized Metrics Registry
# ─────────────────────────────────────────────────────────────────────

class SentinelMetrics:
    """
    Singleton-style container for all Sentinel AI Prometheus metrics.

    Access via the module-level METRICS instance:
        from monitoring.prometheus_metrics import METRICS
    """

    def __init__(self):
        # ── Latency Histograms ─────────────────────────────────────────
        self.vision_latency = _histogram(
            "sentinel_vision_latency_seconds",
            "Vision model inference latency in seconds",
            labels=["model"],
            buckets=LATENCY_BUCKETS,
        )

        self.llm_latency = _histogram(
            "sentinel_llm_latency_seconds",
            "LLM generation latency in seconds",
            labels=["model"],
            buckets=LATENCY_BUCKETS,
        )

        self.pipeline_latency = _histogram(
            "sentinel_pipeline_latency_seconds",
            "End-to-end pipeline latency (vision + RAG + LLM) in seconds",
            labels=["vision_model", "llm_model"],
            buckets=LATENCY_BUCKETS,
        )

        self.rag_latency = _histogram(
            "sentinel_rag_latency_seconds",
            "RAG retrieval latency in seconds",
            labels=["collection"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
        )

        # ── Request Counters ───────────────────────────────────────────
        self.requests_total = _counter(
            "sentinel_requests_total",
            "Total inference requests processed",
            labels=["vision_model", "status"],  # status: success | error
        )

        self.rag_retrievals_total = _counter(
            "sentinel_rag_retrievals_total",
            "Total RAG retrieval operations",
            labels=["outcome"],  # outcome: hit | miss | error
        )

        self.llm_tokens_total = _counter(
            "sentinel_llm_tokens_total",
            "Total LLM tokens generated",
            labels=["model", "type"],  # type: prompt | completion
        )

        self.video_frames_total = _counter(
            "sentinel_video_frames_total",
            "Total video frames processed",
            labels=["model"],
        )

        # ── Gauges (current state) ─────────────────────────────────────
        self.rag_documents_count = _gauge(
            "sentinel_rag_documents_total",
            "Current number of documents in the RAG vector store",
            labels=["collection"],
        )

        self.active_requests = _gauge(
            "sentinel_active_requests",
            "Number of currently in-flight inference requests",
            labels=[],
        )

        self.memory_usage_mb = _gauge(
            "sentinel_memory_usage_mb",
            "Process memory usage in megabytes",
            labels=["component"],  # component: vision | llm | pipeline
        )

        # ── Error Counters ─────────────────────────────────────────────
        self.errors_total = _counter(
            "sentinel_errors_total",
            "Total errors by component and type",
            labels=["component", "error_type"],
        )

    def record_pipeline_run(
        self,
        vision_model: str,
        llm_model: str,
        vision_latency_s: float,
        llm_latency_s: float,
        total_latency_s: float,
        tokens_used: int = 0,
        success: bool = True,
        frames: int = 0,
        rag_hits: int = 0,
        error_type: Optional[str] = None,
    ) -> None:
        """
        Convenience method to record a complete pipeline run.

        Args:
            vision_model: Vision model name (e.g., "llava:7b").
            llm_model: LLM model name (e.g., "llama3.1:8b").
            vision_latency_s: Vision model inference time in seconds.
            llm_latency_s: LLM generation time in seconds.
            total_latency_s: Total end-to-end time in seconds.
            tokens_used: Total LLM tokens generated.
            success: Whether the pipeline completed successfully.
            frames: Number of video frames processed (0 for images).
            rag_hits: Number of RAG chunks retrieved.
            error_type: Error type string if success=False.
        """
        status = "success" if success else "error"

        # Record latencies
        self.vision_latency.labels(model=vision_model).observe(vision_latency_s)
        self.llm_latency.labels(model=llm_model).observe(llm_latency_s)
        self.pipeline_latency.labels(
            vision_model=vision_model, llm_model=llm_model
        ).observe(total_latency_s)

        # Counters
        self.requests_total.labels(vision_model=vision_model, status=status).inc()

        if tokens_used > 0:
            self.llm_tokens_total.labels(model=llm_model, type="completion").inc(tokens_used)

        if frames > 0:
            self.video_frames_total.labels(model=vision_model).inc(frames)

        if rag_hits > 0:
            self.rag_retrievals_total.labels(outcome="hit").inc()
        elif rag_hits == 0 and success:
            # Only count miss if RAG was presumably attempted
            pass

        if not success and error_type:
            self.errors_total.labels(
                component="pipeline", error_type=error_type
            ).inc()

    def update_memory(self) -> float:
        """Measure current process memory and update gauge. Returns MB."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            rss_mb = process.memory_info().rss / (1024 * 1024)
            self.memory_usage_mb.labels(component="process").set(rss_mb)
            return rss_mb
        except Exception:
            return 0.0


# Module-level singleton
METRICS = SentinelMetrics()


# ─────────────────────────────────────────────────────────────────────
# No-op metric fallback
# ─────────────────────────────────────────────────────────────────────

class _NoopMetric:
    """No-op metric for when prometheus_client is unavailable."""

    class _NoopContext:
        def __enter__(self): return self
        def __exit__(self, *args): pass

    def labels(self, **kwargs): return self
    def observe(self, value): pass
    def inc(self, amount=1): pass
    def set(self, value): pass
    def time(self): return self._NoopContext()
