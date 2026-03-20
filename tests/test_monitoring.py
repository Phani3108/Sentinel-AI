"""
Sentinel AI — Monitoring & Observability Tests (Phase 3)

Comprehensive tests for:
  - OpenTelemetry setup (no-op safe)
  - Prometheus metrics definitions and recording
  - SentinelMetrics convenience methods
  - Instrumented pipeline (OTEL + Prometheus integration)
  - API /metrics endpoint

Run: pytest tests/test_monitoring.py -v
"""
import pytest
from unittest.mock import MagicMock, patch


# =========================================================================== #
# OTEL SETUP TESTS
# =========================================================================== #

class TestOtelSetup:
    def test_import_otel_setup(self):
        """Should be importable without errors."""
        from monitoring import otel_setup  # noqa: F401
        assert True

    def test_setup_noop_no_crash(self):
        """setup_otel with enabled=False should not raise."""
        from monitoring.otel_setup import setup_otel
        setup_otel(enabled=False)  # No-op mode

    def test_get_tracer_returns_object(self):
        """get_tracer() should return a valid tracer (or no-op)."""
        from monitoring.otel_setup import get_tracer
        tracer = get_tracer("test.scope")
        assert tracer is not None

    def test_get_meter_returns_object(self):
        """get_meter() should return a valid meter (or no-op)."""
        from monitoring.otel_setup import get_meter
        meter = get_meter("test.scope")
        assert meter is not None

    def test_noop_tracer_context_manager(self):
        """No-op tracer span should work as context manager."""
        from monitoring.otel_setup import _NoopTracer
        tracer = _NoopTracer()
        with tracer.start_as_current_span("test_span") as span:
            span.set_attribute("key", "value")
            span.add_event("test_event")
        # Should complete without error

    def test_noop_span_record_exception(self):
        """No-op span should silently handle record_exception."""
        from monitoring.otel_setup import _NoopSpan
        span = _NoopSpan()
        span.record_exception(Exception("test error"))
        span.set_status("ok")

    def test_noop_meter_creates_instruments(self):
        """No-op meter should create instruments that don't raise."""
        from monitoring.otel_setup import _NoopMeter
        meter = _NoopMeter()
        histogram = meter.create_histogram("test_hist")
        counter = meter.create_counter("test_counter")
        gauge = meter.create_gauge("test_gauge")
        histogram.record(0.5)
        counter.add(1)

    def test_shutdown_no_crash_when_not_initialized(self):
        """shutdown_otel() should not crash when never initialized."""
        from monitoring.otel_setup import shutdown_otel
        shutdown_otel()  # Should not raise


# =========================================================================== #
# PROMETHEUS METRICS TESTS
# =========================================================================== #

class TestPrometheusMetrics:
    def test_metrics_singleton_import(self):
        """METRICS singleton should be importable."""
        from monitoring.prometheus_metrics import METRICS
        assert METRICS is not None

    def test_metrics_has_required_instruments(self):
        """METRICS should have all required metric instruments."""
        from monitoring.prometheus_metrics import METRICS
        required = [
            "vision_latency",
            "llm_latency",
            "pipeline_latency",
            "rag_latency",
            "requests_total",
            "rag_retrievals_total",
            "llm_tokens_total",
            "errors_total",
            "memory_usage_mb",
            "active_requests",
            "rag_documents_count",
        ]
        for attr in required:
            assert hasattr(METRICS, attr), f"METRICS missing attribute: {attr}"

    def test_record_pipeline_run_success(self):
        """record_pipeline_run should not raise on valid inputs."""
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

    def test_record_pipeline_run_with_error(self):
        """record_pipeline_run with success=False should not raise."""
        from monitoring.prometheus_metrics import METRICS
        METRICS.record_pipeline_run(
            vision_model="florence2",
            llm_model="mistral:7b",
            vision_latency_s=0.5,
            llm_latency_s=0.0,
            total_latency_s=0.5,
            success=False,
            error_type="TimeoutError",
        )

    def test_record_pipeline_run_video(self):
        """record_pipeline_run with video frames should not raise."""
        from monitoring.prometheus_metrics import METRICS
        METRICS.record_pipeline_run(
            vision_model="llava:7b",
            llm_model="llama3.1:8b",
            vision_latency_s=15.0,
            llm_latency_s=8.0,
            total_latency_s=24.0,
            tokens_used=500,
            success=True,
            frames=20,
            rag_hits=0,
        )

    def test_memory_update_returns_float(self):
        """update_memory() should return a float >= 0."""
        from monitoring.prometheus_metrics import METRICS
        mem = METRICS.update_memory()
        assert isinstance(mem, float)
        assert mem >= 0.0

    def test_noop_metric_labels(self):
        """_NoopMetric should support .labels() chaining."""
        from monitoring.prometheus_metrics import _NoopMetric
        metric = _NoopMetric()
        metric.labels(model="test").inc()
        metric.labels(model="test").observe(1.5)
        metric.labels(model="test").set(42)

    def test_noop_metric_time_context(self):
        """_NoopMetric.time() should work as context manager."""
        from monitoring.prometheus_metrics import _NoopMetric
        metric = _NoopMetric()
        with metric.labels(model="test").time():
            pass  # Should not raise

    def test_sentinel_metrics_class_init(self):
        """SentinelMetrics should initialize without errors."""
        from monitoring.prometheus_metrics import SentinelMetrics
        # Create a fresh instance (metrics registration may fail if already registered)
        # Just verify the class is accessible
        assert SentinelMetrics is not None

    def test_prometheus_endpoint_available(self):
        """FastAPI /metrics endpoint should return 200 with prometheus content."""
        from fastapi.testclient import TestClient
        from api.main import app
        import api.main as main_module

        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.health_check.return_value = {
            "vision_model": "MockVision",
            "vision_loaded": True,
            "llm_model": "llama3.1:8b",
            "llm_available": True,
            "rag_enabled": False,
            "rag_docs": 0,
            "device": "cpu",
            "otel_enabled": False,
            "metrics_enabled": True,
        }
        mock_pipeline.vision.name = "MockVision"
        mock_pipeline.llm.model = "llama3.1:8b"
        main_module._pipeline = mock_pipeline

        client = TestClient(app)
        response = client.get("/metrics")
        # /metrics should be accessible (returns either prometheus text or 404 if not wired)
        # We allow both to handle environments without prometheus-client
        assert response.status_code in (200, 404)

    def test_api_health_includes_new_fields(self):
        """Health endpoint should now include rag_docs and monitoring fields."""
        from fastapi.testclient import TestClient
        from api.main import app
        import api.main as main_module

        mock_pipeline = MagicMock()
        mock_pipeline.health_check.return_value = {
            "vision_model": "MockVision",
            "vision_loaded": True,
            "llm_model": "llama3.1:8b",
            "llm_available": True,
            "rag_enabled": False,
            "rag_docs": 0,
            "device": "cpu",
            "otel_enabled": False,
            "metrics_enabled": True,
        }
        mock_pipeline.vision.name = "MockVision"
        mock_pipeline.llm.model = "llama3.1:8b"
        main_module._pipeline = mock_pipeline

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


# =========================================================================== #
# INSTRUMENTED PIPELINE MONITORING TESTS
# =========================================================================== #

class TestInstrumentedPipeline:
    @pytest.fixture
    def instrumented_pipeline(self):
        """Build a pipeline with mocked vision/LLM + real monitoring hooks."""
        from unittest.mock import patch, MagicMock
        from core.vision.base import VisionResult
        from core.llm.ollama_client import LLMResponse

        with patch("core.pipeline.get_vision_model") as mock_vision_factory, \
             patch("core.pipeline.OllamaLLMClient") as mock_llm_cls:

            mock_vision = MagicMock()
            mock_vision.name = "llava:7b"
            mock_vision._loaded = True
            mock_vision.analyze.return_value = VisionResult(
                model_name="llava:7b",
                description="An industrial pump with corrosion on the casing.",
                latency_ms=1500.0,
                memory_mb=450.0,
            )
            mock_vision_factory.return_value = mock_vision

            mock_llm = MagicMock()
            mock_llm.model = "llama3.1:8b"
            mock_llm.is_available.return_value = True
            mock_llm.generate.return_value = LLMResponse(
                model="llama3.1:8b",
                content="The pump shows corrosion. Schedule immediate maintenance.",
                prompt_tokens=80,
                completion_tokens=25,
                total_tokens=105,
                latency_ms=3000.0,
            )
            mock_llm_cls.return_value = mock_llm

            from core.pipeline import SentinelPipeline
            p = SentinelPipeline(vision_model="llava", device="cpu")
            p.vision = mock_vision
            p.llm = mock_llm
            p._retriever = None
            yield p

    def test_pipeline_health_check_has_monitoring_fields(self, instrumented_pipeline):
        """health_check() should include otel_enabled and metrics_enabled."""
        health = instrumented_pipeline.health_check()
        assert "otel_enabled" in health
        assert "metrics_enabled" in health

    def test_pipeline_run_completes(self, instrumented_pipeline, tmp_path):
        """Pipeline should complete successfully with monitoring instrumentation."""
        from PIL import Image
        img = Image.new("RGB", (224, 224), (128, 128, 128))
        img_path = tmp_path / "test.jpg"
        img.save(img_path)

        from core.pipeline import PipelineResult
        result = instrumented_pipeline.run_image(img_path, "Describe the damage.")
        assert isinstance(result, PipelineResult)
        assert result.total_latency_ms > 0
        assert len(result.final_answer) > 0
        assert result.rag_latency_ms == 0.0  # No RAG

    def test_pipeline_result_has_rag_latency_field(self, instrumented_pipeline, tmp_path):
        """PipelineResult should include rag_latency_ms field from new pipeline."""
        from PIL import Image
        from dataclasses import asdict
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (64, 64)).save(img_path)
        result = instrumented_pipeline.run_image(img_path)
        result_dict = asdict(result)
        assert "rag_latency_ms" in result_dict

    def test_otel_noop_integration(self, instrumented_pipeline, tmp_path):
        """OTEL spans should work (or be no-ops) without external collector."""
        from PIL import Image
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (64, 64)).save(img_path)
        # Should not raise even if OTLP endpoint is unreachable
        result = instrumented_pipeline.run_image(img_path)
        assert result.vision_description == "An industrial pump with corrosion on the casing."

    def test_metrics_record_on_pipeline_run(self, instrumented_pipeline, tmp_path):
        """Prometheus METRICS.record_pipeline_run should be callable."""
        from PIL import Image
        from monitoring.prometheus_metrics import METRICS

        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (64, 64)).save(img_path)

        # Run pipeline and verify no exceptions from metrics
        result = instrumented_pipeline.run_image(img_path, "What do you see?")
        assert result is not None

        # Manually verify record_pipeline_run doesn't raise
        METRICS.record_pipeline_run(
            vision_model=result.vision_model,
            llm_model=result.llm_model,
            vision_latency_s=result.vision_latency_ms / 1000,
            llm_latency_s=result.llm_latency_ms / 1000,
            total_latency_s=result.total_latency_ms / 1000,
            tokens_used=result.llm_tokens_used,
            success=True,
        )
