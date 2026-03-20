"""
Sentinel AI — Florence-2 Vision Wrapper
Microsoft's Florence-2 model family (0.23B – 0.77B params).

Why Florence-2?
- Tiny — runs on CPU comfortably
- Supports multiple tasks: captioning, OCR, grounding, detection, segmentation
- Apache 2.0 license
- Best model for fine-tuning experiments (small = fast iteration)

Supported task prompts:
    <CAPTION>               — brief caption
    <DETAILED_CAPTION>      — detailed description
    <MORE_DETAILED_CAPTION> — very detailed description
    <OCR>                   — extract text from image
    <OD>                    — object detection (returns bboxes)
    <DENSE_REGION_CAPTION>  — caption every region
    <REGION_PROPOSAL>       — detect objects without labels
    <CAPTION_TO_PHRASE_GROUNDING> — ground a phrase in image
    <REFERRING_EXPRESSION_SEGMENTATION> — segment by description
"""
import logging
from pathlib import Path
from typing import Optional

from PIL import Image

from .base import BaseVisionModel, VisionModelConfig, VisionResult

logger = logging.getLogger(__name__)

# Default HuggingFace model IDs
FLORENCE2_MODELS = {
    "base": "microsoft/Florence-2-base",
    "large": "microsoft/Florence-2-large",
    "base-ft": "microsoft/Florence-2-base-ft",
    "large-ft": "microsoft/Florence-2-large-ft",
}


class Florence2VisionModel(BaseVisionModel):
    """
    Florence-2 wrapper using HuggingFace Transformers.
    Supports all Florence-2 task types via structured prompts.
    """

    def __init__(
        self,
        config: VisionModelConfig,
        florence_variant: str = "base",
        default_task: str = "<DETAILED_CAPTION>",
    ):
        # Auto-resolve model_id from variant name if not explicitly set
        if "/" not in config.model_id:
            config.model_id = FLORENCE2_MODELS.get(config.model_id, FLORENCE2_MODELS["base"])
        super().__init__(config)
        self.default_task = default_task

    @property
    def name(self) -> str:
        return f"Florence-2 ({self.config.model_id.split('/')[-1]})"

    def load(self) -> None:
        """Load Florence-2 model and processor from HuggingFace."""
        from transformers import AutoModelForCausalLM, AutoProcessor
        import torch

        device = self.config.device
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

        logger.info(f"Loading Florence-2 ({self.config.model_id}) on {device}...")
        self._processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)
        self._model.eval()
        logger.info(f"✅ Florence-2 loaded on {device}.")

    def _infer(self, image_path: Path, prompt: str) -> VisionResult:
        """
        Run Florence-2 inference.
        The `prompt` can be either a free-form text (mapped to <DETAILED_CAPTION>)
        or a Florence task string like '<OCR>' or '<OD>'.
        """
        import torch

        image = Image.open(image_path).convert("RGB")

        # Determine task token
        task = prompt if prompt.startswith("<") else self.default_task
        text_input = task

        inputs = self._processor(
            text=text_input,
            images=image,
            return_tensors="pt",
        ).to(self.config.device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=self.config.max_new_tokens,
                num_beams=3,
                do_sample=False,
            )

        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        # Parse structured output from Florence-2's post-processor
        parsed = self._processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.width, image.height),
        )

        return self._to_vision_result(task, parsed, generated_text)

    def _to_vision_result(
        self, task: str, parsed: dict, raw_text: str
    ) -> VisionResult:
        """Convert Florence-2 parsed output → standardized VisionResult."""
        description = ""
        bboxes = None
        tags = []

        if task in ("<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"):
            description = parsed.get(task, raw_text)

        elif task == "<OCR>":
            description = parsed.get("<OCR>", "")
            tags = ["ocr"]

        elif task == "<OD>":
            od_output = parsed.get("<OD>", {})
            labels = od_output.get("labels", [])
            bboxes_raw = od_output.get("bboxes", [])
            tags = list(set(labels))
            bboxes = [
                {"label": lbl, "bbox": bb}
                for lbl, bb in zip(labels, bboxes_raw)
            ]
            description = f"Detected {len(labels)} objects: {', '.join(tags)}"

        elif task == "<DENSE_REGION_CAPTION>":
            drc = parsed.get("<DENSE_REGION_CAPTION>", {})
            labels = drc.get("labels", [])
            description = "; ".join(labels)
            tags = labels

        else:
            description = str(parsed)

        return VisionResult(
            model_name=self.name,
            description=description,
            tags=tags,
            bounding_boxes=bboxes,
            raw_output={"task": task, "parsed": str(parsed)},
        )

    def ocr(self, image: Path) -> str:
        """Convenience method: extract text from image."""
        result = self.analyze(image, prompt="<OCR>")
        return result.description

    def detect_objects(self, image: Path) -> VisionResult:
        """Convenience method: run object detection."""
        return self.analyze(image, prompt="<OD>")
