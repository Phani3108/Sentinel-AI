"""
Sentinel AI — Ollama LLM Client
Communicates with Ollama's REST API for text generation.
Supports: sync inference, async inference, and streaming.
"""
import json
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Iterator, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized output from the Ollama LLM."""
    model: str
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    done: bool = True


class OllamaLLMClient:
    """
    Ollama LLM client with sync + async + streaming support.

    Usage:
        client = OllamaLLMClient(model="llama3.1:8b")
        response = client.generate("Summarize this: ...")
        print(response.content)

    Streaming:
        for token in client.stream("Tell me a story"):
            print(token, end="", flush=True)
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        timeout: int = 120,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or (
            "You are Sentinel AI, a precise and helpful assistant. "
            "Analyze the provided information and respond accurately."
        )
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    # ------------------------------------------------------------------ #
    # Sync API
    # ------------------------------------------------------------------ #

    def generate(self, prompt: str, context: Optional[str] = None) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user prompt
            context: Optional retrieved context (from RAG) prepended to prompt

        Returns:
            LLMResponse with the generated text and token usage
        """
        full_prompt = self._build_prompt(prompt, context)
        payload = self._build_payload(full_prompt)

        t0 = time.perf_counter()
        try:
            response = self._client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()
        except httpx.ConnectError:
            raise RuntimeError(
                "Cannot connect to Ollama. Start it with: ollama serve"
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        data = response.json()

        return LLMResponse(
            model=self.model,
            content=data.get("response", "").strip(),
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            latency_ms=elapsed_ms,
            done=data.get("done", True),
        )

    def stream(self, prompt: str, context: Optional[str] = None) -> Iterator[str]:
        """
        Stream tokens from the LLM as they are generated.

        Yields:
            Token strings (characters / sub-words) in order

        Example:
            for token in client.stream("What is in this image?"):
                print(token, end="", flush=True)
        """
        full_prompt = self._build_prompt(prompt, context)
        payload = dict(self._build_payload(full_prompt), stream=True)

        with self._client.stream(
            "POST",
            f"{self.base_url}/api/generate",
            json=payload,
        ) as response:
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break

    def chat(self, messages: list[dict], context: Optional[str] = None) -> LLMResponse:
        """
        Chat-style interface using Ollama's /api/chat endpoint.

        Args:
            messages: List of {"role": "user" | "assistant", "content": str}
            context: Optional RAG context prepended to the system message

        Returns:
            LLMResponse
        """
        system = self.system_prompt
        if context:
            system += f"\n\nRelevant context:\n{context}"

        all_messages = [{"role": "system", "content": system}] + messages

        payload = {
            "model": self.model,
            "messages": all_messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        t0 = time.perf_counter()
        response = self._client.post(
            f"{self.base_url}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        data = response.json()

        content = data.get("message", {}).get("content", "").strip()
        usage = data.get("usage", {})
        return LLMResponse(
            model=self.model,
            content=content,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------ #
    # Async API
    # ------------------------------------------------------------------ #

    async def agenerate(self, prompt: str, context: Optional[str] = None) -> LLMResponse:
        """Async version of generate()."""
        import httpx as _httpx

        full_prompt = self._build_prompt(prompt, context)
        payload = self._build_payload(full_prompt)

        async with _httpx.AsyncClient(timeout=self.timeout) as client:
            t0 = time.perf_counter()
            response = await client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            response.raise_for_status()
            data = response.json()

        return LLMResponse(
            model=self.model,
            content=data.get("response", "").strip(),
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            latency_ms=elapsed_ms,
        )

    async def astream(self, prompt: str, context: Optional[str] = None) -> AsyncIterator[str]:
        """Async streaming generator of tokens."""
        import httpx as _httpx

        full_prompt = self._build_prompt(prompt, context)
        payload = dict(self._build_payload(full_prompt), stream=True)

        async with _httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json=payload,
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _build_prompt(self, prompt: str, context: Optional[str]) -> str:
        if context:
            return (
                f"System: {self.system_prompt}\n\n"
                f"Context from knowledge base:\n{context}\n\n"
                f"User: {prompt}\nAssistant:"
            )
        return f"System: {self.system_prompt}\n\nUser: {prompt}\nAssistant:"

    def _build_payload(self, full_prompt: str) -> dict:
        return {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            r = self._client.get(f"{self.base_url}/api/tags", timeout=3)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            return any(self.model in m for m in models)
        except Exception:
            return False

    def __repr__(self) -> str:
        return f"OllamaLLMClient(model={self.model}, base_url={self.base_url})"
