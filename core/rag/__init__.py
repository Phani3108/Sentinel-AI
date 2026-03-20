"""
Sentinel AI — RAG (Retrieval-Augmented Generation) Module

Provides document ingestion, embedding, storage, and retrieval
using ChromaDB as the vector store and sentence-transformers for embeddings.

Usage:
    from core.rag import SentinelRetriever

    retriever = SentinelRetriever()
    retriever.ingest_directory("data/docs/")
    docs = retriever.retrieve("What is the maintenance schedule for pump A7?")
    for doc in docs:
        print(doc["text"], doc["score"])
"""
from .retriever import SentinelRetriever
from .embedder import SentinelEmbedder
from .vectorstore import SentinelVectorStore
from .document_loader import load_document, load_directory

__all__ = [
    "SentinelRetriever",
    "SentinelEmbedder",
    "SentinelVectorStore",
    "load_document",
    "load_directory",
]
