"""
Sentinel AI — LLaVA Vision Wrapper
Routes through Ollama's multimodal endpoint (llava:7b or llava:34b).
This is the easiest way to run LLaVA locally — no GPU required for 7B.
"""
import base64
import json
import logging
from pathlib import Path

import httpx

from .base import BaseVisionModel, VisionModelConfig, VisionResult

logger = logging.getLogger(__name__)


class LLaVAVisionModel(BaseVisionModel):
    """
    LLaVA wrapper using Ollama's /api/generate endpoint.

    Ollama handles model loading, quantization (GGUF), and hardware selection
    automatically. This wrapper encodes the image as base64 and sends it
    alongside the prompt to the Ollama multimodal API.

    Supported models (pull with `ollama pull <name>`):
        - llava:7b      (default, ~4GB, CPU-feasible)
        - llava:13b     (~8GB, needs GPU)
        - llava:34b     (~19GB, needs GPU)
        - llava-llama3  (LLaVA on Llama3 backbone)
    """

    def __init__(
        self,
        config: VisionModelConfig,
        ollama_base_url: str = "http://localhost:11434",
    ):
        super().__init__(config)
        self.ollama_base_url = ollama_base_url.rstrip("/")

    @property
    def name(self) -> str:
        return f"LLaVA ({self.config.model_id})"

    def load(self) -> None:
        """
        No local weights to load — Ollama manages the model.
        We verify that Ollama is reachable and the model is available.
        """
        try:
            response = httpx.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            response.raise_for_status()
            tags = response.json()
            available = [m["name"] for m in tags.get("models", [])]
            model = self.config.model_id

            if not any(model in a for a in available):
                logger.warning(
                    f"Model '{model}' not found in Ollama. "
                    f"Run: ollama pull {model}\n"
                    f"Available: {available}"
                )
            else:
                logger.info(f"✅ LLaVA model '{model}' is available in Ollama.")
        except httpx.ConnectError:
            raise RuntimeError(
                "❌ Cannot connect to Ollama. Is it running? Start with: ollama serve"
            )
        self._loaded = True

    def _infer(self, image_path: Path, prompt: str) -> VisionResult:
        """Encode image + send to Ollama multimodal API."""
        image_b64 = self._encode_image(image_path)

        payload = {
            "model": self.config.model_id,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_new_tokens,
            },
        }

        try:
            response = httpx.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
        except httpx.TimeoutException:
            raise TimeoutError(
                f"Ollama request timed out after 120s. "
                f"Consider using a smaller model or increase timeout."
            )

        data = response.json()
        description = data.get("response", "").strip()

        return VisionResult(
            model_name=self.name,
            description=description,
            raw_output=data,
        )

    def stream_infer(self, image_path: Path, prompt: str):
        """
        Generator that yields description tokens as they stream from Ollama.
        Useful for real-time API endpoints.
        """
        image_b64 = self._encode_image(image_path)

        payload = {
            "model": self.config.model_id,
            "prompt": prompt,
            "images": [image_b64],
            "stream": True,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_new_tokens,
            },
        }

        with httpx.stream(
            "POST",
            f"{self.ollama_base_url}/api/generate",
            json=payload,
            timeout=120,
        ) as response:
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    yield token
                    if chunk.get("done"):
                        break

    @staticmethod
    def _encode_image(image_path: Path) -> str:
        """Return base64-encoded image string."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
