"""
Sentinel AI — Moondream2 Vision Wrapper
Ultra-lightweight vision model (1.9B params) from Vikhyat Korrapati.

Why Moondream2?
- Runs fast even on CPU — great for baseline benchmarking
- Supports: image captioning, visual Q&A, object detection
- Simple HuggingFace API
- Good CPU performance on MacBooks and edge devices

HuggingFace: vikhyatk/moondream2
"""
import logging
from pathlib import Path

from PIL import Image

from .base import BaseVisionModel, VisionModelConfig, VisionResult

logger = logging.getLogger(__name__)

MOONDREAM_MODEL_ID = "vikhyatk/moondream2"
MOONDREAM_REVISION = "2025-01-09"  # pin to a stable revision


class MoondreamVisionModel(BaseVisionModel):
    """Moondream2 wrapper — ultra-lightweight, great for CPU inference."""

    @property
    def name(self) -> str:
        return "Moondream2"

    def load(self) -> None:
        """Load Moondream2 model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        device = self.config.device
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

        logger.info(f"Loading Moondream2 on {device}...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            revision=MOONDREAM_REVISION,
            trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            revision=MOONDREAM_REVISION,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation="eager",  # safer cross-platform default
        ).to(device)
        self._model.eval()
        logger.info("✅ Moondream2 loaded.")

    def _infer(self, image_path: Path, prompt: str) -> VisionResult:
        """Run Moondream2 inference — supports caption and VQA prompts."""
        image = Image.open(image_path).convert("RGB")

        # Moondream2 has separate caption and query APIs
        if prompt.lower() in ("caption", "describe", "describe this image"):
            output = self._model.caption(
                images=[image],
                tokenizer=self._tokenizer,
                length="long",
            )
            description = output["caption"] if isinstance(output, dict) else str(output)
        else:
            output = self._model.query(
                images=[image],
                questions=[prompt],
                tokenizer=self._tokenizer,
            )
            if isinstance(output, dict):
                description = output.get("answer", str(output))
            elif isinstance(output, list):
                description = output[0] if output else ""
            else:
                description = str(output)

        return VisionResult(
            model_name=self.name,
            description=description,
            raw_output={"prompt": prompt, "output": str(output)},
        )

    def caption(self, image: Path, length: str = "long") -> str:
        """Convenience method: get a caption for the image."""
        result = self.analyze(image, prompt="caption")
        return result.description

    def ask(self, image: Path, question: str) -> str:
        """Convenience method: ask a question about the image."""
        result = self.analyze(image, prompt=question)
        return result.description
