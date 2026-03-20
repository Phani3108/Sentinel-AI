"""
Sentinel AI — Document Embedder

Wraps sentence-transformers to produce dense embeddings
for documents and queries. Supports batch processing and
multiple embedding models.

Models (in order of recommendation):
  - "all-MiniLM-L6-v2"   — fast, 384-dim, great for English
  - "all-mpnet-base-v2"  — higher quality, 768-dim
  - "nomic-ai/nomic-embed-text-v1" — long context (8192 tokens)
"""
import logging
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class SentinelEmbedder:
    """
    Wraps a sentence-transformers model to embed text.

    Args:
        model_name: HuggingFace model name or local path.
        device: "cpu", "cuda", or "mps".
        batch_size: Number of texts per batch.
        normalize: Whether to L2-normalize embeddings.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None  # lazy-loaded

    def _ensure_loaded(self) -> None:
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                )
                logger.info(f"✅ Embedding model loaded: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )

    @property
    def dimension(self) -> int:
        """Return the embedding dimension of this model."""
        self._ensure_loaded()
        return self._model.get_sentence_embedding_dimension()

    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Embed a single text or a list of texts.

        Args:
            text: A string, or list of strings to embed.

        Returns:
            np.ndarray of shape (n_texts, dim) or (dim,) for a single string.
        """
        self._ensure_loaded()
        if isinstance(text, str):
            texts = [text]
            single = True
        else:
            texts = list(text)
            single = False

        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 50,
        )

        return embeddings[0] if single else embeddings

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch and return as list of lists (for ChromaDB).

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        arr = self.embed(texts)
        return arr.tolist()

    def similarity(self, a: Union[str, np.ndarray], b: Union[str, np.ndarray]) -> float:
        """
        Compute cosine similarity between two texts or embeddings.

        Args:
            a, b: Either text strings or pre-computed embedding vectors.

        Returns:
            Cosine similarity score in [-1, 1].
        """
        if isinstance(a, str):
            a = self.embed(a)
        if isinstance(b, str):
            b = self.embed(b)

        a = np.array(a, dtype=np.float32)
        b = np.array(b, dtype=np.float32)

        # Cosine similarity (normalize=True means vectors are already unit norm)
        dot = float(np.dot(a, b))
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))

        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def __repr__(self) -> str:
        loaded = self._model is not None
        return f"SentinelEmbedder(model={self.model_name}, device={self.device}, loaded={loaded})"
