"""
Sentinel AI — RAG Retriever

High-level interface that combines the embedder, vector store,
and document loader into a single retrieval API.

Features:
  - Ingest single files or entire directories
  - Query with semantic similarity
  - Optional BM25-style keyword reranking
  - Source-filtered queries
  - Deduplication

Usage:
    from core.rag import SentinelRetriever

    retriever = SentinelRetriever()

    # Ingest your documents
    retriever.ingest_directory("data/docs/")

    # Query
    docs = retriever.retrieve("pump maintenance schedule")
    for doc in docs:
        print(f"[{doc['score']:.2f}] {doc['text'][:100]}")
        print(f"  Source: {doc['metadata'].get('source')}")
"""
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .embedder import SentinelEmbedder
from .vectorstore import SentinelVectorStore
from .document_loader import load_document, load_directory, Document

logger = logging.getLogger(__name__)


class SentinelRetriever:
    """
    End-to-end RAG retriever for Sentinel AI.

    Combines:
    - SentinelEmbedder  → dense embeddings
    - SentinelVectorStore → ChromaDB storage and similarity search
    - document_loader   → file ingestion and chunking

    Args:
        embedding_model: sentence-transformers model name.
        device: "cpu", "cuda", or "mps".
        collection_name: ChromaDB collection name.
        persist_directory: Local path for ChromaDB storage.
        top_k: Default number of results to retrieve.
        score_threshold: Minimum similarity score (0–1) to include a result.
        use_http: Connect to remote ChromaDB over HTTP.
        host: Remote ChromaDB host (when use_http=True).
        port: Remote ChromaDB port (when use_http=True).
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        collection_name: str = "sentinel_docs",
        persist_directory: str = "./data/chroma_db",
        top_k: int = 5,
        score_threshold: float = 0.0,
        use_http: bool = False,
        host: str = "localhost",
        port: int = 8000,
    ):
        self.top_k = top_k
        self.score_threshold = score_threshold

        logger.info(f"Initializing SentinelRetriever (model={embedding_model})")

        self.embedder = SentinelEmbedder(
            model_name=embedding_model,
            device=device,
        )
        self.vectorstore = SentinelVectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
            host=host,
            port=port,
            use_http=use_http,
        )

        logger.info(
            f"✅ SentinelRetriever ready — "
            f"{self.vectorstore.count} documents in collection"
        )

    # ────────────────────────────────────────────
    # Ingestion
    # ────────────────────────────────────────────

    def ingest(self, document: Document) -> str:
        """
        Ingest a single Document chunk into the vector store.

        Returns:
            The document's ID.
        """
        embedding = self.embedder.embed(document.text).tolist()
        ids = self.vectorstore.add(
            texts=[document.text],
            embeddings=[embedding],
            metadatas=[document.metadata],
            ids=[document.doc_id],
        )
        return ids[0]

    def ingest_documents(self, documents: List[Document]) -> List[str]:
        """
        Batch-ingest a list of Document chunks.

        Returns:
            List of document IDs.
        """
        if not documents:
            return []

        texts = [doc.text for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [doc.doc_id for doc in documents]

        # Batch embed all texts at once
        logger.info(f"Embedding {len(texts)} chunks...")
        embeddings = self.embedder.embed_batch(texts)

        inserted_ids = self.vectorstore.add(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info(f"✅ Ingested {len(inserted_ids)} chunks")
        return inserted_ids

    def ingest_file(
        self,
        path: Union[str, Path],
        chunk_size: int = 512,
        overlap: int = 64,
    ) -> List[str]:
        """
        Load a file, chunk it, and ingest all chunks.

        Args:
            path: Path to document file (.txt, .md, .pdf, .docx, .json).
            chunk_size: Characters per chunk.
            overlap: Character overlap between chunks.

        Returns:
            List of inserted document IDs.
        """
        docs = load_document(path, chunk_size=chunk_size, overlap=overlap)
        return self.ingest_documents(docs)

    def ingest_directory(
        self,
        directory: Union[str, Path],
        chunk_size: int = 512,
        overlap: int = 64,
        glob: str = "**/*",
        exclude_patterns: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Recursively load and ingest all documents in a directory.

        Args:
            directory: Directory path.
            chunk_size: Characters per chunk.
            overlap: Character overlap between chunks.
            glob: Glob pattern.
            exclude_patterns: File patterns to skip.

        Returns:
            List of all inserted document IDs.
        """
        docs = load_directory(
            directory,
            glob=glob,
            chunk_size=chunk_size,
            overlap=overlap,
            exclude_patterns=exclude_patterns,
        )
        if not docs:
            logger.warning(f"No documents found in '{directory}'")
            return []
        return self.ingest_documents(docs)

    # ────────────────────────────────────────────
    # Retrieval
    # ────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        source_filter: Optional[str] = None,
        rerank: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant document chunks for a query.

        Args:
            query: The search query string.
            top_k: Number of results (overrides instance default).
            score_threshold: Minimum similarity score.
            source_filter: Restrict results to a specific source filename.
            rerank: Apply keyword-boost reranking on top of vector search.

        Returns:
            List of dicts: {id, text, metadata, distance, score}
            Sorted by score descending.
        """
        k = top_k or self.top_k
        threshold = score_threshold if score_threshold is not None else self.score_threshold

        if self.vectorstore.count == 0:
            logger.warning("Vector store is empty — no documents to retrieve from.")
            return []

        # Embed the query
        query_embedding = self.embedder.embed(query).tolist()

        # Metadata filter
        where = None
        if source_filter:
            where = {"source": {"$contains": source_filter}}

        # Vector similarity search
        results = self.vectorstore.query(
            query_embedding=query_embedding,
            top_k=k * 2 if rerank else k,  # fetch extra for reranking
            where=where,
        )

        # Filter by score threshold
        results = [r for r in results if r["score"] >= threshold]

        # Optional keyword reranking
        if rerank and results:
            results = self._keyword_rerank(query, results)

        # Return top-k
        return results[:k]

    def retrieve_as_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        separator: str = "\n---\n",
    ) -> Optional[str]:
        """
        Retrieve documents and format as a single context string
        ready to be injected into an LLM prompt.

        Returns:
            Formatted context string, or None if no results.
        """
        docs = self.retrieve(query, top_k=top_k)
        if not docs:
            return None

        parts = []
        for doc in docs:
            source = doc["metadata"].get("source", "unknown")
            score = doc["score"]
            parts.append(f"[Source: {Path(source).name} | Score: {score:.2f}]\n{doc['text']}")

        return separator.join(parts)

    # ────────────────────────────────────────────
    # Management
    # ────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all documents from the vector store."""
        self.vectorstore.reset()
        logger.info("Vector store has been reset.")

    @property
    def document_count(self) -> int:
        """Return the total number of stored document chunks."""
        return self.vectorstore.count

    # ────────────────────────────────────────────
    # Internal helpers
    # ────────────────────────────────────────────

    @staticmethod
    def _keyword_rerank(query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Simple keyword-based reranking on top of vector similarity.

        Boosts documents that contain exact query terms.
        Returns results sorted by adjusted score.
        """
        query_terms = set(re.findall(r"\w+", query.lower()))
        if not query_terms:
            return results

        for doc in results:
            doc_terms = set(re.findall(r"\w+", doc["text"].lower()))
            overlap = len(query_terms & doc_terms)
            keyword_boost = overlap / max(len(query_terms), 1) * 0.15  # max +15% boost
            doc["score"] = min(1.0, doc["score"] + keyword_boost)

        return sorted(results, key=lambda x: x["score"], reverse=True)

    def __repr__(self) -> str:
        return (
            f"SentinelRetriever("
            f"model={self.embedder.model_name}, "
            f"docs={self.document_count}, "
            f"top_k={self.top_k})"
        )
