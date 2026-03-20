"""
Sentinel AI — Pipeline Integration Tests
Deep tests for config, pipeline orchestration, and video utilities.

Run: pytest tests/test_pipeline.py -v
"""
import pytest
import tempfile
from pathlib import Path
from dataclasses import asdict
from unittest.mock import MagicMock, patch, AsyncMock


# =========================================================================== #
# FIXTURES
# =========================================================================== #

@pytest.fixture
def sample_image(tmp_path) -> Path:
    """Create a 224x224 test image."""
    from PIL import Image
    img = Image.new("RGB", (224, 224), color=(120, 80, 200))
    path = tmp_path / "test.jpg"
    img.save(path, "JPEG")
    return path


@pytest.fixture
def sample_video(tmp_path) -> Path:
    """Create a minimal 5-frame synthetic video using OpenCV."""
    try:
        import cv2
        import numpy as np
        path = tmp_path / "test_video.avi"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(path), fourcc, 5.0, (64, 64))
        for i in range(10):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            frame[:, :, i % 3] = (i * 25) % 255
            writer.write(frame)
        writer.release()
        return path
    except ImportError:
        pytest.skip("OpenCV not installed — skipping video tests")


@pytest.fixture
def mock_vision_result():
    from core.vision.base import VisionResult
    return VisionResult(
        model_name="MockVision",
        description="A test image with a red ball on a blue background.",
        tags=["ball", "blue", "red"],
        latency_ms=42.5,
        memory_mb=15.0,
    )


@pytest.fixture
def mock_llm_response():
    from core.llm.ollama_client import LLMResponse
    return LLMResponse(
        model="llama3.1:8b",
        content="The image shows a red ball on a blue background. This appears to be a test image.",
        prompt_tokens=30,
        completion_tokens=20,
        total_tokens=50,
        latency_ms=120.0,
    )


# =========================================================================== #
# CONFIG TESTS
# =========================================================================== #

class TestConfig:
    def test_settings_defaults(self):
        """Settings should have sensible defaults even without .env."""
        from core.config import Settings
        s = Settings()
        assert s.ollama_base_url == "http://localhost:11434"
        assert s.device == "cpu"
        assert s.llm_temperature == 0.1
        assert s.llm_max_tokens == 1024
        assert s.rag_top_k == 5
        assert s.api_port == 8080

    def test_settings_are_cached(self):
        """get_settings() should return the same object (lru_cache)."""
        from core.config import get_settings
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_settings_override_via_env(self, monkeypatch):
        """Environment variables should override defaults."""
        monkeypatch.setenv("DEVICE", "mps")
        monkeypatch.setenv("LLM_MAX_TOKENS", "2048")
        from core.config import Settings
        s = Settings()
        assert s.device == "mps"
        assert s.llm_max_tokens == 2048


# =========================================================================== #
# IMAGE UTILS DEEP TESTS
# =========================================================================== #

class TestImageUtils:
    def test_load_image_rgb(self, sample_image):
        from core.utils.image_utils import load_image
        img = load_image(sample_image, convert="RGB")
        assert img.mode == "RGB"
        assert img.size == (224, 224)

    def test_load_image_from_bytes(self, sample_image):
        from core.utils.image_utils import load_image
        with open(sample_image, "rb") as f:
            raw = f.read()
        img = load_image(raw)
        assert img.mode == "RGB"

    def test_resize_named_size(self, sample_image):
        from core.utils.image_utils import load_image, resize_image
        img = load_image(sample_image)
        resized = resize_image(img, size="thumbnail")
        assert resized.size[0] <= 224
        assert resized.size[1] <= 224

    def test_resize_exact_no_aspect(self, sample_image):
        from core.utils.image_utils import load_image, resize_image
        img = load_image(sample_image)
        resized = resize_image(img, size=(100, 200), keep_aspect=False)
        assert resized.size == (100, 200)

    def test_image_to_bytes_jpeg(self, sample_image):
        from core.utils.image_utils import load_image, image_to_bytes
        img = load_image(sample_image)
        data = image_to_bytes(img, format="JPEG", quality=80)
        assert isinstance(data, bytes)
        assert len(data) > 0
        assert data[:2] == b'\xff\xd8'  # JPEG magic bytes

    def test_batch_images(self, tmp_path):
        from PIL import Image
        from core.utils.image_utils import batch_images
        imgs = []
        for i in range(3):
            img = Image.new("RGB", (100, 100), color=(i * 80, i * 80, i * 80))
            p = tmp_path / f"img_{i}.jpg"
            img.save(p)
            imgs.append(p)
        result = batch_images(imgs, size="thumbnail")
        assert len(result) == 3

    def test_batch_images_skips_bad_file(self, tmp_path):
        from core.utils.image_utils import batch_images
        bad_path = tmp_path / "nonexistent.jpg"
        result = batch_images([bad_path])
        assert result == []

    def test_normalize_returns_tensor(self, sample_image):
        from core.utils.image_utils import load_image, normalize_image
        img = load_image(sample_image)
        tensor = normalize_image(img)
        assert tensor.shape == (3, 224, 224)
        # After normalization, values should be roughly in [-3, 3] range
        assert float(tensor.min()) > -5.0
        assert float(tensor.max()) < 5.0


# =========================================================================== #
# VIDEO UTILS TESTS
# =========================================================================== #

class TestVideoUtils:
    def test_extract_frames_creates_files(self, sample_video, tmp_path):
        """extract_frames should create JPEG files on disk."""
        from core.utils.video_utils import extract_frames
        frames = extract_frames(
            sample_video,
            output_dir=tmp_path / "frames",
            fps=5.0,
            max_frames=5,
        )
        assert len(frames) > 0
        for frame in frames:
            assert frame.image_path.exists()
            assert frame.width == 64
            assert frame.height == 64

    def test_extract_frames_respects_max(self, sample_video, tmp_path):
        """max_frames limit should be respected."""
        from core.utils.video_utils import extract_frames
        frames = extract_frames(sample_video, output_dir=tmp_path, fps=10.0, max_frames=3)
        assert len(frames) <= 3

    def test_frames_timestamps_monotonic(self, sample_video, tmp_path):
        """Frame timestamps should be non-decreasing."""
        from core.utils.video_utils import extract_frames
        frames = extract_frames(sample_video, output_dir=tmp_path, fps=2.0, max_frames=10)
        timestamps = [f.timestamp_sec for f in frames]
        assert all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))

    def test_frames_to_narration_input(self, sample_video, tmp_path):
        """frames_to_narration_input should produce list of dicts."""
        from core.utils.video_utils import extract_frames, frames_to_narration_input
        frames = extract_frames(sample_video, output_dir=tmp_path, fps=5.0, max_frames=3)
        narration_input = frames_to_narration_input(frames)
        assert isinstance(narration_input, list)
        for item in narration_input:
            assert "timestamp" in item
            assert "image_path" in item

    def test_extract_frames_bad_video(self, tmp_path):
        """Should raise IOError for non-video file."""
        try:
            import cv2  # noqa: F401
        except ImportError:
            pytest.skip("OpenCV not installed — skipping video tests")
        from core.utils.video_utils import extract_frames
        fake = tmp_path / "fake.mp4"
        fake.write_bytes(b"not a video")
        with pytest.raises(IOError):
            extract_frames(fake, output_dir=tmp_path, fps=1.0)

    def test_video_frame_dataclass(self, sample_video, tmp_path):
        """VideoFrame should have all expected fields."""
        from core.utils.video_utils import extract_frames
        frames = extract_frames(sample_video, output_dir=tmp_path, fps=5.0, max_frames=1)
        if frames:
            f = frames[0]
            assert isinstance(f.index, int)
            assert isinstance(f.timestamp_sec, float)
            assert isinstance(f.image_path, Path)
            assert f.width > 0
            assert f.height > 0


# =========================================================================== #
# OLLAMA LLM CLIENT TESTS (mocked network)
# =========================================================================== #

class TestOllamaClient:
    def test_client_instantiation(self):
        from core.llm.ollama_client import OllamaLLMClient
        client = OllamaLLMClient(model="llama3.1:8b", base_url="http://localhost:11434")
        assert client.model == "llama3.1:8b"
        assert client.base_url == "http://localhost:11434"

    def test_build_prompt_with_context(self):
        from core.llm.ollama_client import OllamaLLMClient
        client = OllamaLLMClient()
        prompt = client._build_prompt("What is this?", "Blue background, red ball")
        assert "What is this?" in prompt
        assert "Blue background, red ball" in prompt

    def test_build_prompt_without_context(self):
        from core.llm.ollama_client import OllamaLLMClient
        client = OllamaLLMClient()
        prompt = client._build_prompt("Hello?", None)
        assert "Hello?" in prompt
        assert "Context" not in prompt

    def test_build_payload(self):
        from core.llm.ollama_client import OllamaLLMClient
        client = OllamaLLMClient(model="mistral:7b", temperature=0.5, max_tokens=512)
        payload = client._build_payload("test prompt")
        assert payload["model"] == "mistral:7b"
        assert payload["options"]["temperature"] == 0.5
        assert payload["options"]["num_predict"] == 512

    def test_is_available_when_offline(self):
        """is_available() should return False when Ollama is not running."""
        from core.llm.ollama_client import OllamaLLMClient
        client = OllamaLLMClient(base_url="http://localhost:99999")
        assert client.is_available() is False

    @patch("core.llm.ollama_client.httpx.Client.post")
    def test_generate_mock(self, mock_post, mock_llm_response):
        from core.llm.ollama_client import OllamaLLMClient
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": "A red ball on blue background.",
            "prompt_eval_count": 30,
            "eval_count": 15,
            "done": True,
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = OllamaLLMClient()
        result = client.generate("What do you see?")
        assert "red ball" in result.content.lower() or len(result.content) > 0
        assert result.done is True

    @patch("core.llm.ollama_client.httpx.Client.post")
    def test_chat_mock(self, mock_post):
        from core.llm.ollama_client import OllamaLLMClient
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "This is a test."},
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = OllamaLLMClient()
        result = client.chat([{"role": "user", "content": "Hello"}])
        assert isinstance(result.content, str)
        assert len(result.content) > 0


# =========================================================================== #
# PIPELINE TESTS (mocked vision + LLM)
# =========================================================================== #

class TestSentinelPipeline:
    @pytest.fixture
    def pipeline(self):
        """Build a SentinelPipeline with mocked dependencies."""
        with patch("core.pipeline.get_vision_model") as mock_vision_factory, \
             patch("core.pipeline.OllamaLLMClient") as mock_llm_cls:

            mock_vision = MagicMock()
            mock_vision.name = "MockVision"
            mock_vision._loaded = False
            mock_vision_factory.return_value = mock_vision

            mock_llm = MagicMock()
            mock_llm.model = "llama3.1:8b"
            mock_llm.is_available.return_value = True
            mock_llm_cls.return_value = mock_llm

            from core.pipeline import SentinelPipeline
            p = SentinelPipeline(vision_model="llava", device="cpu")
            p.vision = mock_vision
            p.llm = mock_llm
            p._retriever = None
            yield p

    def test_pipeline_repr(self, pipeline):
        r = repr(pipeline)
        assert "SentinelPipeline" in r
        assert "MockVision" in r

    def test_health_check_structure(self, pipeline):
        health = pipeline.health_check()
        required_keys = {
            "vision_model", "vision_loaded", "llm_model", "llm_available",
            "rag_enabled", "device",
            # Phase 3 additions:
            "rag_docs", "otel_enabled", "metrics_enabled",
        }
        assert required_keys == set(health.keys())

    def test_build_llm_prompt_structure(self):
        from core.pipeline import SentinelPipeline
        from core.vision.base import VisionResult
        vr = VisionResult(model_name="Test", description="A dog running in a park.")
        prompt = SentinelPipeline._build_llm_prompt("Is the dog happy?", vr)
        assert "A dog running in a park." in prompt
        assert "Is the dog happy?" in prompt
        assert "VISION MODEL OUTPUT" in prompt

    def test_run_image_returns_pipeline_result(self, pipeline, sample_image):
        from core.vision.base import VisionResult
        from core.llm.ollama_client import LLMResponse
        from core.pipeline import PipelineResult

        pipeline.vision.analyze.return_value = VisionResult(
            model_name="MockVision",
            description="Red ball on blue background.",
            latency_ms=50.0,
            memory_mb=10.0,
        )
        pipeline.llm.generate.return_value = LLMResponse(
            model="llama3.1:8b",
            content="This appears to be a test image with colorful elements.",
            prompt_tokens=20,
            completion_tokens=15,
            total_tokens=35,
            latency_ms=100.0,
        )

        result = pipeline.run_image(sample_image, "Describe the scene.")
        assert isinstance(result, PipelineResult)
        assert result.input_type == "image"
        assert "Red ball" in result.vision_description
        assert len(result.final_answer) > 0
        assert result.total_latency_ms > 0
        assert result.llm_tokens_used == 35

    def test_pipeline_result_is_dataclass(self, pipeline, sample_image):
        from core.vision.base import VisionResult
        from core.llm.ollama_client import LLMResponse

        pipeline.vision.analyze.return_value = VisionResult(
            model_name="Test", description="Scene", latency_ms=10.0, memory_mb=5.0
        )
        pipeline.llm.generate.return_value = LLMResponse(
            model="test", content="Answer", latency_ms=50.0
        )
        result = pipeline.run_image(sample_image)
        # Should be convertible to dict (dataclass)
        result_dict = asdict(result)
        assert isinstance(result_dict, dict)
        assert "vision_description" in result_dict
        assert "final_answer" in result_dict
        assert "total_latency_ms" in result_dict


# =========================================================================== #
# FASTAPI ENDPOINT TESTS
# =========================================================================== #

class TestFastAPIEndpoints:
    @pytest.fixture
    def test_client(self):
        """Create a FastAPI test client with mocked pipeline."""
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
            "device": "cpu",
        }
        mock_pipeline.vision.name = "MockVision"
        mock_pipeline.llm.model = "llama3.1:8b"

        main_module._pipeline = mock_pipeline
        return TestClient(app)

    def test_health_endpoint(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "vision_model" in data
        assert "device" in data

    def test_models_endpoint(self, test_client):
        response = test_client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "vision_models" in data
        assert "llava" in data["vision_models"]
        assert "florence2" in data["vision_models"]
        assert "moondream" in data["vision_models"]

    def test_analyze_image_endpoint(self, test_client):
        from core.pipeline import PipelineResult
        from io import BytesIO
        from PIL import Image

        # Inject mock pipeline with run_image
        from api import main as main_module
        from datetime import datetime, timezone

        main_module._pipeline.run_image.return_value = PipelineResult(
            input_path="/tmp/test.jpg",
            input_type="image",
            prompt="What do you see?",
            vision_model="MockVision",
            vision_description="A colorful test image.",
            vision_latency_ms=45.0,
            vision_memory_mb=10.0,
            llm_model="llama3.1:8b",
            final_answer="This is a colorful test image.",
            llm_latency_ms=100.0,
            llm_tokens_used=25,
            total_latency_ms=150.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Create a dummy image to upload
        img_buf = BytesIO()
        Image.new("RGB", (64, 64), color=(100, 150, 200)).save(img_buf, format="JPEG")
        img_buf.seek(0)

        response = test_client.post(
            "/analyze/image",
            files={"file": ("test.jpg", img_buf, "image/jpeg")},
            data={"prompt": "What do you see?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "vision_description" in data
        assert "final_answer" in data
        assert "total_latency_ms" in data
        assert data["total_latency_ms"] > 0

    def test_openapi_docs_available(self, test_client):
        """FastAPI docs should be accessible."""
        response = test_client.get("/docs")
        assert response.status_code == 200

    def test_openapi_schema(self, test_client):
        """OpenAPI JSON schema should be accessible."""
        response = test_client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "paths" in schema
        assert "/health" in schema["paths"]
        assert "/analyze/image" in schema["paths"]
