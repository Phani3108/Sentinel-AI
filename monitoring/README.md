# Sentinel AI — Monitoring & Observability Setup

This guide explains how to deploy and use the full monitoring stack for Sentinel AI.

---

## Architecture

```
Sentinel AI FastAPI App
    │
    ├── OpenTelemetry SDK
    │       ├── Traces  → OTLP → Jaeger (http://localhost:16686)
    │       └── Metrics → OTLP → Collector → Prometheus
    │
    ├── prometheus-client /metrics endpoint
    │       └── Prometheus scrapes every 15s (http://localhost:9090)
    │               └── Grafana dashboards (http://localhost:3000)
    │
    └── Structured logs → stdout (configurable via LOG_LEVEL env var)
```

---

## Quick Start (Docker Compose)

```bash
# Start all monitoring services
cd docker/
docker compose up -d prometheus grafana jaeger chromadb ollama

# Wait for services to start (~30s)
sleep 30

# Start the API (local dev)
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload

# Access services
open http://localhost:9090   # Prometheus
open http://localhost:3000   # Grafana (admin/admin)
open http://localhost:16686  # Jaeger traces
```

---

## Grafana Setup

### Import Dashboards

1. Open Grafana: http://localhost:3000 (login: admin / admin)
2. Go to **Dashboards → Import**
3. Upload `monitoring/dashboards/sentinel_inference.json`
4. Select your Prometheus datasource
5. Click **Import**

### Dashboard Panels

| Panel | Metric | Description |
|-------|--------|-------------|
| Vision Latency | `sentinel_vision_latency_seconds` | p50/p95/p99 per model |
| LLM Latency | `sentinel_llm_latency_seconds` | p50/p95 per model |
| Request Throughput | `sentinel_requests_total` | Requests/sec |
| Error Rate | `sentinel_errors_total` | % errors (threshold alerts) |
| Memory Usage | `sentinel_memory_usage_mb` | RSS memory over time |
| Pipeline Latency | `sentinel_pipeline_latency_seconds` | End-to-end p95 |
| Token Throughput | `sentinel_llm_tokens_total` | Tokens/sec |
| RAG Documents | `sentinel_rag_documents_total` | Docs in vector store |
| RAG Latency | `sentinel_rag_latency_seconds` | Retrieval p95 |

---

## Prometheus Metrics Reference

All metrics are exposed at `GET /metrics` on the Sentinel AI API.

### Histograms

| Metric | Labels | Description |
|--------|--------|-------------|
| `sentinel_vision_latency_seconds` | `model` | Vision model inference time |
| `sentinel_llm_latency_seconds` | `model` | LLM generation time |
| `sentinel_pipeline_latency_seconds` | `vision_model`, `llm_model` | End-to-end time |
| `sentinel_rag_latency_seconds` | `collection` | RAG retrieval time |

### Counters

| Metric | Labels | Description |
|--------|--------|-------------|
| `sentinel_requests_total` | `vision_model`, `status` | Total requests |
| `sentinel_rag_retrievals_total` | `outcome` | RAG retrieval outcomes |
| `sentinel_llm_tokens_total` | `model`, `type` | LLM tokens generated |
| `sentinel_video_frames_total` | `model` | Video frames processed |
| `sentinel_errors_total` | `component`, `error_type` | Errors by type |

### Gauges

| Metric | Labels | Description |
|--------|--------|-------------|
| `sentinel_rag_documents_total` | `collection` | Docs in vector store |
| `sentinel_active_requests` | — | In-flight requests |
| `sentinel_memory_usage_mb` | `component` | Memory usage (MB) |

---

## OpenTelemetry Traces

Traces are sent to Jaeger via OTLP gRPC at `http://localhost:4317`.

### Instrumented Operations

| Span Name | Component | Key Attributes |
|-----------|-----------|----------------|
| `vision_inference` | Vision Layer | `model.name`, `image.size` |
| `llm_generate` | LLM Layer | `model.name`, `prompt.length`, `tokens.total` |
| `rag_retrieve` | RAG Layer | `collection`, `top_k`, `docs_returned` |
| `pipeline_run` | Pipeline | `input_type`, `total_latency_ms` |
| `api_request` | API | `endpoint`, `status_code` |

### Viewing Traces in Jaeger

1. Open http://localhost:16686
2. Select service: **sentinel-ai**
3. Search by operation name or trace ID
4. View waterfall view of vision → RAG → LLM spans

---

## Alerting

Configure Grafana alerts on these critical thresholds:

```yaml
# Vision latency p95 > 10s
sum(rate(sentinel_vision_latency_seconds_bucket{le="10"}[5m])) < 0.95

# Error rate > 5%
rate(sentinel_errors_total[5m]) / rate(sentinel_requests_total[5m]) > 0.05

# Memory > 90% of 8GB
sentinel_memory_usage_mb{component="process"} > 7372
```

---

## Python SDK Usage

### Record a pipeline run

```python
from monitoring.prometheus_metrics import METRICS

METRICS.record_pipeline_run(
    vision_model="llava:7b",
    llm_model="llama3.1:8b",
    vision_latency_s=2.3,
    llm_latency_s=5.1,
    total_latency_s=7.8,
    tokens_used=320,
    success=True,
    rag_hits=3,
)
```

### Use OpenTelemetry spans

```python
from monitoring.otel_setup import get_tracer

tracer = get_tracer("sentinel.pipeline")

with tracer.start_as_current_span("vision_inference") as span:
    span.set_attribute("model.name", "llava:7b")
    result = vision_model.analyze(image_path, prompt)
    span.set_attribute("latency_ms", result.latency_ms)
```

---

*Last updated: 2026-03-20 | Sentinel AI v0.2.0*
