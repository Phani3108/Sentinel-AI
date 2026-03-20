"""
Sentinel AI — Vision Model Registry
Provides a unified factory for instantiating vision models by name.
"""
from .base import BaseVisionModel, VisionModelConfig, VisionResult
from .llava import LLaVAVisionModel
from .florence import Florence2VisionModel
from .moondream import MoondreamVisionModel
from .phi3v import Phi3VisionModel
from .internvl import InternVL2VisionModel

__all__ = [
    "BaseVisionModel",
    "VisionModelConfig",
    "VisionResult",
    "LLaVAVisionModel",
    "Florence2VisionModel",
    "MoondreamVisionModel",
    "Phi3VisionModel",
    "InternVL2VisionModel",
    "get_vision_model",
]

# Registry: name → (class, default_model_id)
_REGISTRY = {
    "llava": (LLaVAVisionModel, "llava:7b"),
    "llava:7b": (LLaVAVisionModel, "llava:7b"),
    "llava:13b": (LLaVAVisionModel, "llava:13b"),
    "llava:34b": (LLaVAVisionModel, "llava:34b"),
    "florence2": (Florence2VisionModel, "microsoft/Florence-2-base"),
    "florence2-large": (Florence2VisionModel, "microsoft/Florence-2-large"),
    "moondream": (MoondreamVisionModel, "vikhyatk/moondream2"),
    "moondream2": (MoondreamVisionModel, "vikhyatk/moondream2"),
    "phi3v": (Phi3VisionModel, "microsoft/Phi-3.5-vision-instruct"),
    "phi3.5-vision": (Phi3VisionModel, "microsoft/Phi-3.5-vision-instruct"),
    "internvl": (InternVL2VisionModel, "OpenGVLab/InternVL2-2B"),
    "internvl2-1b": (InternVL2VisionModel, "OpenGVLab/InternVL2-1B"),
    "internvl2-2b": (InternVL2VisionModel, "OpenGVLab/InternVL2-2B"),
    "internvl2-4b": (InternVL2VisionModel, "OpenGVLab/InternVL2-4B"),
}


def get_vision_model(
    name: str,
    device: str = "cpu",
    **kwargs,
) -> BaseVisionModel:
    """
    Factory function: instantiate a vision model by name.

    Args:
        name: Model name from the registry (e.g., 'llava', 'florence2', 'moondream')
        device: 'cpu', 'cuda', or 'mps'
        **kwargs: Extra args passed to VisionModelConfig

    Returns:
        Unloaded BaseVisionModel instance (call .analyze() to auto-load)

    Example:
        model = get_vision_model("florence2", device="cpu")
        result = model.analyze("image.jpg", "<DETAILED_CAPTION>")
    """
    name_lower = name.lower()
    if name_lower not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ValueError(
            f"Unknown vision model: '{name}'. "
            f"Available: {available}"
        )

    cls, default_model_id = _REGISTRY[name_lower]
    config = VisionModelConfig(
        model_id=kwargs.pop("model_id", default_model_id),
        device=device,
        **kwargs,
    )

    # LLaVA needs special ollama_base_url kwarg
    if cls == LLaVAVisionModel:
        ollama_url = kwargs.pop("ollama_base_url", "http://localhost:11434")
        return cls(config, ollama_base_url=ollama_url)

    return cls(config)
