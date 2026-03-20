"""
Sentinel AI — Cloud Client (Phase 6)
OpenAI GPT-4V wrapper for cloud routing.
"""
import base64
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Price per 1K tokens (USD) — update as pricing changes
_GPT4O_INPUT_PRICE  = 0.005   # $5 / 1M tokens
_GPT4O_OUTPUT_PRICE = 0.015   # $15 / 1M tokens


class CloudClient:
    """
    OpenAI GPT-4o / GPT-4V client for cloud-side inference.
    Falls back gracefully if OPENAI_API_KEY is not configured.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        max_tokens: int = 1024,
    ):
        import os
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.max_tokens = max_tokens

        if not self.api_key:
            logger.warning(
                "OPENAI_API_KEY not set — cloud routing disabled. "
                "Set the env var to enable GPT-4o routing."
            )

    def is_available(self) -> bool:
        return bool(self.api_key)

    def analyze_image(self, image_path: str, prompt: str) -> dict:
        """
        Send an image + prompt to GPT-4o and return a standardised result dict.

        Returns:
            dict with keys: description, answer, model, tokens_used, cost_usd, latency_ms
        """
        if not self.is_available():
            raise RuntimeError("No OPENAI_API_KEY configured — cannot route to cloud.")

        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required: pip install httpx")

        # Encode image to base64
        img_bytes = Path(image_path).read_bytes()
        img_b64 = base64.b64encode(img_bytes).decode()

        # Detect MIME type
        suffix = Path(image_path).suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                    ".webp": "image/webp", ".gif": "image/gif"}
        mime = mime_map.get(suffix, "image/jpeg")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": self.max_tokens,
        }

        t_start = time.perf_counter()
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        latency_ms = (time.perf_counter() - t_start) * 1000
        resp.raise_for_status()

        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        input_tokens  = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        total_tokens  = usage.get("total_tokens", 0)
        cost_usd = (input_tokens * _GPT4O_INPUT_PRICE + output_tokens * _GPT4O_OUTPUT_PRICE) / 1000

        logger.info(
            f"[Cloud] GPT-4o response: {total_tokens} tokens, "
            f"${cost_usd:.4f}, {latency_ms:.0f}ms"
        )

        return {
            "description": "",    # GPT-4o combines vision+text in one pass
            "answer": content,
            "model": self.model,
            "tokens_used": total_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost_usd, 6),
            "latency_ms": round(latency_ms, 1),
        }
