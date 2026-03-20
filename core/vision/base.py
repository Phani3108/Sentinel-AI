"""
Sentinel AI — Vision Model Base Class
All vision model wrappers implement this interface.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
import time


@dataclass
class VisionResult:
    """Standardized output from any vision model."""
    model_name: str
    description: str                     # Primary natural language description
    tags: list[str] = field(default_factory=list)  # Detected labels / categories
    confidence: Optional[float] = None  # Model confidence if available
    bounding_boxes: Optional[list] = None  # [{"label": str, "bbox": [x,y,w,h]}]
    raw_output: Optional[dict] = None   # Full model output for debugging
    latency_ms: float = 0.0            # Inference wall-clock time
    memory_mb: float = 0.0             # Peak memory during inference


@dataclass
class VisionModelConfig:
    """Configuration for a vision model."""
    model_id: str                       # HF model ID or Ollama model name
    device: str = "cpu"                 # cpu | cuda | mps
    max_new_tokens: int = 512
    temperature: float = 0.1
    load_in_4bit: bool = False          # QLoRA quantization
    load_in_8bit: bool = False


class BaseVisionModel(ABC):
    """
    Abstract base class for all Sentinel AI vision model wrappers.
    
    Each concrete implementation wraps a specific model (LLaVA, Florence-2, etc.)
    and exposes a unified interface: analyze(image, prompt) → VisionResult.
    """

    def __init__(self, config: VisionModelConfig):
        self.config = config
        self._model = None
        self._processor = None
        self._loaded = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""
        ...

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory. Called lazily on first use."""
        ...

    @abstractmethod
    def _infer(self, image_path: Path, prompt: str) -> VisionResult:
        """Core inference — subclasses implement this."""
        ...

    def analyze(
        self,
        image: Union[str, Path],
        prompt: str = "Describe this image in detail.",
    ) -> VisionResult:
        """
        Public interface — load model if needed, run inference, return result.
        Automatically measures latency and memory usage.
        """
        import psutil, os

        image_path = Path(image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if not self._loaded:
            self.load()
            self._loaded = True

        # Track memory before inference
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        t0 = time.perf_counter()
        result = self._infer(image_path, prompt)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        mem_after = process.memory_info().rss / 1024 / 1024
        result.latency_ms = elapsed_ms
        result.memory_mb = mem_after - mem_before
        return result

    def unload(self) -> None:
        """Release model from memory."""
        import gc, torch
        self._model = None
        self._processor = None
        self._loaded = False
        gc.collect()
        if self.config.device == "cuda":
            torch.cuda.empty_cache()

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "unloaded"
        return f"{self.__class__.__name__}(model={self.config.model_id}, device={self.config.device}, status={status})"
