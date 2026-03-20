"""
Sentinel AI — Phi-3.5 Vision Wrapper
Microsoft's Phi-3.5 Vision Instruct model (4.2B params).

Why Phi-3.5 Vision?
- Strong reasoning + vision in a compact 4.2B package
- Runs on CPU and Apple MPS (Metal)
- Instruction-tuned with chat template support
- Great at documents, charts, multi-image tasks

HuggingFace: microsoft/Phi-3.5-vision-instruct
"""
import logging
from pathlib import Path
from typing import Union

from PIL import Image

from .base import BaseVisionModel, VisionModelConfig, VisionResult

logger = logging.getLogger(__name__)

PHI3V_MODEL_ID = "microsoft/Phi-3.5-vision-instruct"


class Phi3VisionModel(BaseVisionModel):
    """Phi-3.5 Vision wrapper with chat-template support."""

    @property
    def name(self) -> str:
        return "Phi-3.5-Vision"

    def load(self) -> None:
        """Load Phi-3.5 Vision model and processor."""
        from transformers import AutoModelForCausalLM, AutoProcessor
        import torch

        device = self.config.device
        # Phi-3.5 Vision requires flash_attention_2 on CUDA; use eager elsewhere
        attn_impl = "flash_attention_2" if device == "cuda" else "eager"
        dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32

        logger.info(f"Loading Phi-3.5-Vision on {device}...")
        self._processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
            num_crops=4,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=dtype,
            _attn_implementation=attn_impl,
        )
        self._model.eval()
        logger.info("✅ Phi-3.5-Vision loaded.")

    def _infer(self, image_path: Path, prompt: str) -> VisionResult:
        """Run Phi-3.5 Vision inference using chat template."""
        import torch

        image = Image.open(image_path).convert("RGB")

        # Phi-3.5 Vision uses a structured chat format with image placeholders
        messages = [
            {"role": "user", "content": f"<|image_1|>\n{prompt}"}
        ]
        text = self._processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._processor(
            text=text,
            images=[image],
            return_tensors="pt",
        ).to(self.config.device)

        generate_ids = self._model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            do_sample=self.config.temperature > 0,
            eos_token_id=self._processor.tokenizer.eos_token_id,
        )

        # Remove the input tokens from the output
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        description = self._processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return VisionResult(
            model_name=self.name,
            description=description,
            raw_output={"prompt": prompt},
        )
