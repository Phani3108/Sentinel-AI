"""
Sentinel AI — Vision Pipeline Tests
Tests the vision model wrappers and full pipeline.

Run: pytest tests/test_vision.py -v
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def sample_image(tmp_path) -> Path:
    """Create a minimal test image (1x1 white pixel JPG)."""
    from PIL import Image
    img = Image.new("RGB", (224, 224), color=(200, 200, 200))
    path = tmp_path / "test_image.jpg"
    img.save(path, "JPEG")
    return path


@pytest.fixture
def vision_config():
    from core.vision.base import VisionModelConfig
    return VisionModelConfig(model_id="vikhyatk/moondream2", device="cpu")


# --------------------------------------------------------------------------- #
# Base Model Tests
# --------------------------------------------------------------------------- #

def test_vision_result_defaults():
    from core.vision.base import VisionResult
    result = VisionResult(model_name="TestModel", description="A test image")
    assert result.model_name == "TestModel"
    assert result.description == "A test image"
    assert result.latency_ms == 0.0
    assert result.tags == []


def test_vision_config_defaults():
    from core.vision.base import VisionModelConfig
    config = VisionModelConfig(model_id="test/model")
    assert config.device == "cpu"
    assert config.max_new_tokens == 512
    assert config.temperature == 0.1
    assert config.load_in_4bit is False


# --------------------------------------------------------------------------- #
# Registry Tests
# --------------------------------------------------------------------------- #

def test_get_vision_model_invalid():
    from core.vision import get_vision_model
    with pytest.raises(ValueError, match="Unknown vision model"):
        get_vision_model("nonexistent_model_xyz")


def test_get_vision_model_llava():
    from core.vision import get_vision_model, LLaVAVisionModel
    model = get_vision_model("llava", device="cpu")
    assert isinstance(model, LLaVAVisionModel)
    assert model.config.device == "cpu"
    assert not model._loaded


def test_get_vision_model_florence2():
    from core.vision import get_vision_model, Florence2VisionModel
    model = get_vision_model("florence2", device="cpu")
    assert isinstance(model, Florence2VisionModel)


def test_get_vision_model_moondream():
    from core.vision import get_vision_model, MoondreamVisionModel
    model = get_vision_model("moondream", device="cpu")
    assert isinstance(model, MoondreamVisionModel)


def test_all_registry_keys():
    """All registry entries should instantiate without error."""
    from core.vision import get_vision_model
    for name in ["llava", "florence2", "moondream", "phi3v", "internvl"]:
        model = get_vision_model(name, device="cpu")
        assert model is not None
        assert not model._loaded  # lazy loading — not loaded yet


# --------------------------------------------------------------------------- #
# Image Utils Tests
# --------------------------------------------------------------------------- #

def test_load_image(sample_image):
    from core.utils.image_utils import load_image
    image = load_image(sample_image)
    assert image.mode == "RGB"


def test_resize_image(sample_image):
    from core.utils.image_utils import load_image, resize_image
    image = load_image(sample_image)
    resized = resize_image(image, size=(100, 100), keep_aspect=False)
    assert resized.size == (100, 100)


def test_image_to_base64(sample_image):
    from core.utils.image_utils import image_to_base64
    import base64
    b64 = image_to_base64(sample_image)
    # Should be valid base64
    decoded = base64.b64decode(b64)
    assert len(decoded) > 0


def test_get_image_info(sample_image):
    from core.utils.image_utils import get_image_info
    info = get_image_info(sample_image)
    assert info["width"] == 224
    assert info["height"] == 224
    assert info["mode"] == "RGB"
    assert "file_size_kb" in info


def test_load_image_missing_file():
    from core.utils.image_utils import load_image
    with pytest.raises(Exception):
        load_image("/nonexistent/path/image.jpg")


# --------------------------------------------------------------------------- #
# LLava Wrapper Tests (mocked Ollama)
# --------------------------------------------------------------------------- #

def test_llava_load_checks_ollama():
    """LLaVA load() should raise RuntimeError if Ollama is not running."""
    from core.vision.llava import LLaVAVisionModel
    from core.vision.base import VisionModelConfig
    import httpx

    config = VisionModelConfig(model_id="llava:7b", device="cpu")
    model = LLaVAVisionModel(config, ollama_base_url="http://localhost:99999")  # bad port

    with pytest.raises((RuntimeError, Exception)):
        model.load()


def test_llava_encode_image(sample_image):
    from core.vision.llava import LLaVAVisionModel
    from core.vision.base import VisionModelConfig
    import base64

    config = VisionModelConfig(model_id="llava:7b", device="cpu")
    model = LLaVAVisionModel(config)
    b64 = model._encode_image(sample_image)
    decoded = base64.b64decode(b64)
    assert len(decoded) > 0


def test_llava_analyze_missing_image():
    from core.vision.llava import LLaVAVisionModel
    from core.vision.base import VisionModelConfig

    config = VisionModelConfig(model_id="llava:7b", device="cpu")
    model = LLaVAVisionModel(config)
    model._loaded = True  # Skip load()

    with pytest.raises(FileNotFoundError):
        model.analyze("/nonexistent/image.jpg")


# --------------------------------------------------------------------------- #
# Pipeline Orchestrator Tests (mocked vision + LLM)
# --------------------------------------------------------------------------- #

def test_pipeline_build_llm_prompt():
    from core.pipeline import SentinelPipeline
    from core.vision.base import VisionResult

    result = VisionResult(model_name="Test", description="A sunny park.")
    prompt = SentinelPipeline._build_llm_prompt("What is the mood?", result)
    assert "A sunny park." in prompt
    assert "What is the mood?" in prompt


def test_pipeline_health_check():
    """Pipeline health_check should return a dict with required keys."""
    with patch("core.pipeline.get_vision_model") as mock_vision, \
         patch("core.pipeline.OllamaLLMClient") as mock_llm:

        mock_vision.return_value = MagicMock(name="MockVision", _loaded=False)
        mock_vision.return_value.name = "MockVision"
        mock_llm.return_value = MagicMock(model="llama3.1:8b")
        mock_llm.return_value.is_available.return_value = False

        from core.pipeline import SentinelPipeline
        pipeline = SentinelPipeline.__new__(SentinelPipeline)
        pipeline.vision = mock_vision.return_value
        pipeline.llm = mock_llm.return_value
        pipeline.device = "cpu"
        pipeline.enable_rag = False
        pipeline._retriever = None
        # Phase 3 monitoring attrs
        pipeline._tracer = None
        pipeline._metrics = None

        health = pipeline.health_check()
        assert "vision_model" in health
        assert "llm_model" in health
        assert "device" in health
        # Phase 3 additions
        assert "otel_enabled" in health
        assert "metrics_enabled" in health
        assert "rag_docs" in health
