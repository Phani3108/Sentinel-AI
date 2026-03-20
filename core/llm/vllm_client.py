"""
Sentinel AI — vLLM Inference Client
Phase 8: High Availability & Scaling

Provides a high-throughput, OpenAI-compatible engine client for replacing
Ollama in heavily loaded production environments. Supports PagedAttention
and continuous batching via vLLM.
"""
import os
import logging
from typing import Optional

try:
    from openai import OpenAI
    openai_available = True
except ImportError:
    openai_available = False

logger = logging.getLogger(__name__)

class VLLMClient:
    """Wrapper around vLLM's OpenAI-compatible HTTP server."""
    
    def __init__(self, base_url: str = None, api_key: str = "EMPTY", model: str = None):
        self.base_url = base_url or os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")
        self.api_key = api_key
        self.model = model or os.getenv("VLLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
        
        if not openai_available:
            logger.warning("vLLM Client disabled (`openai` package not installed).")
            self.client = None
            return

        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            logger.info(f"Initialized vLLM Client pointing at {self.base_url}")
        except Exception as e:
            logger.error(f"Failed to init vLLM Client: {e}")
            self.client = None

    def is_available(self) -> bool:
        return self.client is not None

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Synchronous generation using vLLM."""
        if not self.is_available():
            raise RuntimeError("vLLM Client is not available.")
            
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"vLLM Generation Error: {e}")
            raise e

    def stream(self, prompt: str, system_prompt: Optional[str] = None):
        """Streaming generator using vLLM."""
        if not self.is_available():
            raise RuntimeError("vLLM Client is not available.")
            
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=1024,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"vLLM Streaming Error: {e}")
            raise e

    @property
    def model_name(self) -> str:
        return self.model
