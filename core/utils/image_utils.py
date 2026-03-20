"""
Sentinel AI — Image Preprocessing Utilities
Handles image loading, resizing, normalization, and format conversion.
Provides a consistent interface for all vision model wrappers.
"""
import io
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# Common image sizes
SIZES = {
    "thumbnail": (224, 224),
    "small": (384, 384),
    "standard": (448, 448),
    "large": (672, 672),
    "hd": (1344, 1344),
}


def load_image(
    source: Union[str, Path, bytes],
    convert: str = "RGB",
) -> Image.Image:
    """
    Load an image from a file path or raw bytes.

    Args:
        source: File path, Path object, or raw bytes
        convert: PIL mode to convert to ('RGB', 'L', 'RGBA')

    Returns:
        PIL Image object
    """
    if isinstance(source, bytes):
        image = Image.open(io.BytesIO(source))
    else:
        image = Image.open(Path(source))

    return image.convert(convert)


def resize_image(
    image: Image.Image,
    size: Union[str, Tuple[int, int]] = "standard",
    keep_aspect: bool = True,
) -> Image.Image:
    """
    Resize an image to the target dimensions.

    Args:
        image: Input PIL Image
        size: Named size ('thumbnail', 'small', 'standard', 'large', 'hd')
              or (width, height) tuple
        keep_aspect: If True, uses thumbnail (keeps aspect ratio);
                     if False, stretches to exact dimensions

    Returns:
        Resized PIL Image
    """
    if isinstance(size, str):
        target = SIZES.get(size, SIZES["standard"])
    else:
        target = size

    if keep_aspect:
        image = ImageOps.contain(image, target)
    else:
        image = image.resize(target, Image.LANCZOS)

    return image


def normalize_image(
    image: Image.Image,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
):
    """
    Normalize a PIL Image to a torch tensor using ImageNet stats.

    Returns:
        torch.Tensor of shape (3, H, W)
    """
    import torch
    import numpy as np

    arr = np.array(image).astype(np.float32) / 255.0  # [0, 1]
    arr = (arr - np.array(mean)) / np.array(std)
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
    return tensor


def image_to_base64(image_path: Union[str, Path]) -> str:
    """Return base64-encoded JPEG string of an image file."""
    import base64
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_to_bytes(
    image: Image.Image,
    format: str = "JPEG",
    quality: int = 90,
) -> bytes:
    """Convert PIL Image to bytes (useful for API payloads)."""
    buf = io.BytesIO()
    image.save(buf, format=format, quality=quality)
    return buf.getvalue()


def get_image_info(image_path: Union[str, Path]) -> dict:
    """
    Extract basic image metadata.

    Returns:
        dict with width, height, mode, format, file_size_kb
    """
    path = Path(image_path)
    image = Image.open(path)
    stat = path.stat()
    return {
        "path": str(path),
        "filename": path.name,
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "format": image.format,
        "file_size_kb": round(stat.st_size / 1024, 2),
        "aspect_ratio": round(image.width / image.height, 3),
    }


def batch_images(
    image_paths: list[Union[str, Path]],
    size: Union[str, Tuple[int, int]] = "standard",
):
    """
    Load and resize a list of images for batch inference.

    Returns:
        List of PIL Images
    """
    images = []
    for p in image_paths:
        try:
            img = load_image(p)
            img = resize_image(img, size)
            images.append(img)
        except Exception as e:
            logger.warning(f"Failed to load image {p}: {e}")
    return images
