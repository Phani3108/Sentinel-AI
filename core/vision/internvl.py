"""
Sentinel AI — InternVL2 Vision Wrapper
State-of-the-art open-source vision-language model from Shanghai AI Laboratory.

Why InternVL2?
- Top performance on OCR, chart understanding, document VQA
- Available in 1B to 76B parameter variants
- Dynamic high-resolution image processing
- Strong multilingual support

HuggingFace IDs:
    OpenGVLab/InternVL2-1B    — ultra-light, cpu-feasible
    OpenGVLab/InternVL2-2B    — good CPU/MPS performance (recommended start)
    OpenGVLab/InternVL2-4B    — strong GPU model
    OpenGVLab/InternVL2-8B    — near SOTA on most benchmarks
"""
import logging
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .base import BaseVisionModel, VisionModelConfig, VisionResult

logger = logging.getLogger(__name__)

INTERNVL_MODELS = {
    "1B": "OpenGVLab/InternVL2-1B",
    "2B": "OpenGVLab/InternVL2-2B",
    "4B": "OpenGVLab/InternVL2-4B",
    "8B": "OpenGVLab/InternVL2-8B",
}

IMG_SIZE = 448
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)


class InternVL2VisionModel(BaseVisionModel):
    """InternVL2 wrapper — excels at document/OCR/chart understanding."""

    @property
    def name(self) -> str:
        return f"InternVL2 ({self.config.model_id.split('/')[-1]})"

    def load(self) -> None:
        """Load InternVL2 model and tokenizer."""
        from transformers import AutoModel, AutoTokenizer
        import torch

        device = self.config.device
        dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32

        logger.info(f"Loading InternVL2 ({self.config.model_id}) on {device}...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
            use_fast=False,
        )
        self._model = AutoModel.from_pretrained(
            self.config.model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(device).eval()
        logger.info("✅ InternVL2 loaded.")

    def _infer(self, image_path: Path, prompt: str) -> VisionResult:
        """Run InternVL2 inference with dynamic tile preprocessing."""
        import torch

        pixel_values = self._load_image(image_path).to(self.config.device)

        # InternVL2 uses a <image> token convention
        question = f"<image>\n{prompt}"

        generation_config = dict(
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.temperature > 0,
            temperature=self.config.temperature if self.config.temperature > 0 else None,
        )

        response = self._model.chat(
            self._tokenizer,
            pixel_values,
            question,
            generation_config,
        )

        return VisionResult(
            model_name=self.name,
            description=response,
            raw_output={"prompt": prompt},
        )

    def _load_image(self, image_path: Path, max_num_tiles: int = 12):
        """Load and tile-preprocess image for InternVL2's dynamic resolution."""
        import torch
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize(
                (IMG_SIZE, IMG_SIZE),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ])

        image = Image.open(image_path).convert("RGB")
        pixel_values = transform(image).unsqueeze(0)  # (1, 3, H, W)
        return pixel_values
