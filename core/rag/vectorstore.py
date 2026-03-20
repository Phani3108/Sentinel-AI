"""
Sentinel AI — ChromaDB Vector Store Interface

Provides a clean abstraction over ChromaDB for storing and querying
document embeddings. Supports both persistent (local) and HTTP (remote)
ChromaDB backends.

Two operation modes:
  1. Persistent (local): uses chromadb.PersistentClient (best for dev)
  2. HTTP (remote): uses chromadb.HttpClient (Docker/production)
"""
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default persistent storage path
DEFAULT_PERSIST_DIR = "./data/chroma_db"


class SentinelVectorStore:
    """
    ChromaDB-backed vector store for Sentinel AI.

    Supports:
    - Adding documents with embeddings and metadata
    - Similarity search (top-k retrieval)
    - Collection reset / deletion
    - Both persistent local and HTTP remote backends

    Args:
        collection_name: Name of the ChromaDB collection.
        persist_directory: Local path for persistent storage (local mode).
        host: ChromaDB server host (HTTP mode).
        port: ChromaDB server port (HTTP mode).
        use_http: If True, use HTTP client (Docker deployment).
    """

    def __init__(
        self,
        collection_name: str = "sentinel_docs",
        persist_directory: str = DEFAULT_PERSIST_DIR,
        host: str = "localhost",
        port: int = 8000,
        use_http: bool = False,
    ):
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.host = host
        self.port = port
        self.use_http = use_http
        self._client = None
        self._collection = None
        self._initialize()

    def _initialize(self) -> None:
        """Create ChromaDB client and get/create the collection."""
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb not installed. Run: pip install chromadb"
            )

        if self.use_http:
            logger.info(f"Connecting to ChromaDB at {self.host}:{self.port}")
            self._client = chromadb.HttpClient(host=self.host, port=self.port)
        else:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using persistent ChromaDB at {self.persist_directory}")
            self._client = chromadb.PersistentClient(path=str(self.persist_directory))

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )
        count = self._collection.count()
        logger.info(
            f"✅ Collection '{self.collection_name}' ready — {count} documents"
        )

    @property
    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self._collection.count()

    def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            texts: The raw text content of each document chunk.
            embeddings: Pre-computed embeddings (list of float lists).
            metadatas: Optional list of metadata dicts (source, page, etc.).
            ids: Optional document IDs. Auto-generated UUIDs if not provided.

        Returns:
            List of document IDs that were inserted.
        """
        if not texts:
            return []

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        # Default empty metadata — pass None if all empty (ChromaDB rejects empty {})
        if metadatas is None:
            metadatas_arg = None
        else:
            # Ensure metadata values are ChromaDB-compatible (str/int/float/bool)
            clean_metadatas = []
            for meta in metadatas:
                if not meta:
                    clean_metadatas.append(None)
                    continue
                clean = {}
                for k, v in meta.items():
                    if isinstance(v, (str, int, float, bool)):
                        clean[k] = v
                    else:
                        clean[k] = str(v)
                clean_metadatas.append(clean if clean else None)
            metadatas_arg = clean_metadatas

        self._collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas_arg,
            ids=ids,
        )
        logger.info(f"Added {len(texts)} documents to '{self.collection_name}'")
        return ids

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most similar documents.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            where: Optional ChromaDB metadata filter.

        Returns:
            List of dicts with keys: id, text, metadata, distance, score.
        """
        if self._collection.count() == 0:
            return []

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self._collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        docs = []
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            # Convert distance to similarity score (cosine: 0=identical, 2=opposite)
            score = 1.0 - (dist / 2.0)
            docs.append({
                "id": results["ids"][0][i],
                "text": doc,
                "metadata": meta,
                "distance": dist,
                "score": round(score, 4),
            })

        return docs

    def delete(self, ids: List[str]) -> None:
        """Delete documents by their IDs."""
        if ids:
            self._collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from '{self.collection_name}'")

    def reset(self) -> None:
        """Delete and recreate the collection (clears all data)."""
        logger.warning(f"Resetting collection '{self.collection_name}'")
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Collection '{self.collection_name}' has been reset")

    def get_all_ids(self) -> List[str]:
        """Return all document IDs in the collection."""
        if self._collection.count() == 0:
            return []
        result = self._collection.get(include=[])
        return result["ids"]

    def __repr__(self) -> str:
        mode = "http" if self.use_http else "persistent"
        return (
            f"SentinelVectorStore("
            f"collection={self.collection_name}, "
            f"mode={mode}, "
            f"count={self.count})"
        )
