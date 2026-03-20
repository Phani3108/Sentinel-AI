"""
Sentinel AI — RAG Integration Tests (Phase 2)

Comprehensive tests for:
  - Document loading and chunking
  - SentinelEmbedder (sentence-transformers)
  - SentinelVectorStore (ChromaDB)
  - SentinelRetriever (end-to-end)
  - Pipeline integration with enable_rag=True

Run: pytest tests/test_rag.py -v
"""
import pytest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch


# =========================================================================== #
# FIXTURES
# =========================================================================== #

@pytest.fixture
def sample_docs_dir(tmp_path) -> Path:
    """Create a temporary directory with sample .md and .txt files."""
    d = tmp_path / "docs"
    d.mkdir()

    (d / "maintenance.md").write_text(
        "# Pump Maintenance\n\n"
        "Pumps should be inspected every 2000 hours.\n\n"
        "Replace seals if wear exceeds 0.3mm.\n\n"
        "Use Mobil Grease XHP 222 for bearings.",
        encoding="utf-8",
    )

    (d / "safety.txt").write_text(
        "Always wear PPE when working near pumps.\n"
        "LOTO procedures are mandatory before maintenance.\n"
        "High-visibility vest required in Zone A.",
        encoding="utf-8",
    )

    (d / "policy.md").write_text(
        "# AI Policy\n\n"
        "Sensitive data must be processed locally.\n\n"
        "PII includes faces, ID documents, and medical records.\n\n"
        "Cloud routing is only allowed for PUBLIC classification.",
        encoding="utf-8",
    )

    return d


@pytest.fixture
def sample_text_file(tmp_path) -> Path:
    """Create a single text file with content."""
    f = tmp_path / "test_doc.txt"
    f.write_text(
        "Sentinel AI is a local multimodal AI stack.\n\n"
        "It supports LLaVA, Florence-2, and InternVL for vision.\n\n"
        "The LLM backend uses Ollama for local inference.\n\n"
        "RAG retrieval is provided by ChromaDB and sentence-transformers.",
        encoding="utf-8",
    )
    return f


# =========================================================================== #
# DOCUMENT LOADER TESTS
# =========================================================================== #

class TestDocumentLoader:
    def test_load_text_file(self, sample_text_file):
        """Should load a .txt file and return Document chunks."""
        from core.rag.document_loader import load_document
        docs = load_document(sample_text_file)
        assert len(docs) > 0
        for doc in docs:
            assert isinstance(doc.text, str)
            assert len(doc.text) > 0
            assert "source" in doc.metadata

    def test_load_markdown_file(self, tmp_path):
        """Should load a .md file correctly."""
        md_file = tmp_path / "test.md"
        md_file.write_text("# Title\n\nThis is some content.\n\nAnother paragraph here.", encoding="utf-8")
        from core.rag.document_loader import load_document
        docs = load_document(md_file)
        assert len(docs) > 0
        full_text = " ".join(d.text for d in docs)
        assert "Title" in full_text or "content" in full_text

    def test_file_not_found(self, tmp_path):
        """Should raise FileNotFoundError for missing file."""
        from core.rag.document_loader import load_document
        with pytest.raises(FileNotFoundError):
            load_document(tmp_path / "nonexistent.txt")

    def test_unsupported_format(self, tmp_path):
        """Should raise ValueError for unsupported file types."""
        f = tmp_path / "image.png"
        f.write_bytes(b"fake png content")
        from core.rag.document_loader import load_document
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_document(f)

    def test_chunk_size_respected(self, tmp_path):
        """Chunks should not massively exceed chunk_size."""
        long_text = "This is a sentence about pumps. " * 100
        f = tmp_path / "long.txt"
        f.write_text(long_text, encoding="utf-8")
        from core.rag.document_loader import load_document
        docs = load_document(f, chunk_size=200, overlap=20)
        # All chunks should be reasonable in size
        for doc in docs:
            # Allow some flexibility due to overlap
            assert len(doc.text) < 1000, f"Chunk too large: {len(doc.text)} chars"

    def test_load_directory(self, sample_docs_dir):
        """Should load all supported files from a directory."""
        from core.rag.document_loader import load_directory
        docs = load_directory(sample_docs_dir)
        assert len(docs) > 0
        sources = {d.metadata.get("source", "") for d in docs}
        assert any("maintenance" in s for s in sources)
        assert any("safety" in s for s in sources)
        assert any("policy" in s for s in sources)

    def test_load_empty_directory(self, tmp_path):
        """Empty directory should return empty list."""
        from core.rag.document_loader import load_directory
        docs = load_directory(tmp_path)
        assert docs == []

    def test_document_has_uuid(self, sample_text_file):
        """Each document chunk should have a unique doc_id."""
        from core.rag.document_loader import load_document
        docs = load_document(sample_text_file)
        ids = [d.doc_id for d in docs]
        assert len(ids) == len(set(ids)), "Document IDs are not unique"

    def test_chunk_overlap(self, tmp_path):
        """Consecutive chunks should share content from overlap."""
        # Create text that will definitely be split into multiple chunks
        text = ("Sentence about pumps. " * 30 + "\n\n") * 3
        f = tmp_path / "overlap_test.txt"
        f.write_text(text, encoding="utf-8")
        from core.rag.document_loader import load_document, chunk_text
        chunks = chunk_text(text, chunk_size=200, overlap=50, source="test")
        if len(chunks) > 1:
            # The second chunk should start with content from the end of the first
            prev_tail = chunks[0].text[-50:]
            # At least some words from prev_tail should appear in chunk[1]
            prev_words = set(prev_tail.split())
            next_words = set(chunks[1].text.split())
            overlap_words = prev_words & next_words
            assert len(overlap_words) > 0


# =========================================================================== #
# EMBEDDER TESTS (mocked sentence-transformers)
# =========================================================================== #

class TestSentinelEmbedder:
    @pytest.fixture
    def mock_embedder(self):
        """Return an embedder with mocked sentence-transformers."""
        from core.rag.embedder import SentinelEmbedder
        embedder = SentinelEmbedder(model_name="all-MiniLM-L6-v2", device="cpu")

        # Mock the internal ST model
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384

        import numpy as np
        def fake_encode(texts, **kwargs):
            n = len(texts)
            arr = np.random.randn(n, 384).astype(np.float32)
            # Normalize
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            return arr / norms

        mock_model.encode.side_effect = fake_encode
        embedder._model = mock_model
        return embedder

    def test_embedder_repr(self, mock_embedder):
        """repr should include model name and loaded status."""
        r = repr(mock_embedder)
        assert "SentinelEmbedder" in r
        assert "all-MiniLM-L6-v2" in r

    def test_embed_single_text(self, mock_embedder):
        """Should return 1D array for single string input."""
        import numpy as np
        result = mock_embedder.embed("test text")
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert result.shape[0] == 384

    def test_embed_batch(self, mock_embedder):
        """Should return 2D array for list of strings."""
        import numpy as np
        texts = ["first text", "second text", "third text"]
        result = mock_embedder.embed(texts)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 384)

    def test_embed_batch_returns_lists(self, mock_embedder):
        """embed_batch should return list of float lists."""
        result = mock_embedder.embed_batch(["hello", "world"])
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert all(isinstance(x, float) for x in result[0])

    def test_dimension_property(self, mock_embedder):
        """dimension property should return model embedding size."""
        assert mock_embedder.dimension == 384

    def test_similarity_identical_texts(self, mock_embedder):
        """Identical embeddings should have similarity close to 1.0."""
        import numpy as np
        # Mock to return same vector for both calls
        vec = np.ones(384, dtype=np.float32)
        vec = vec / np.linalg.norm(vec)
        mock_embedder.embed = MagicMock(return_value=vec)
        score = mock_embedder.similarity("same text", "same text")
        assert abs(score - 1.0) < 1e-5

    def test_similarity_returns_float(self, mock_embedder):
        """similarity() should return a float in [-1, 1]."""
        score = mock_embedder.similarity("pumps are machines", "water flows through pipes")
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0


# =========================================================================== #
# VECTOR STORE TESTS (mocked ChromaDB)
# =========================================================================== #

class TestSentinelVectorStore:
    @pytest.fixture
    def vectorstore(self, tmp_path):
        """Return a real (persistent) ChromaDB vector store."""
        try:
            import chromadb  # noqa: F401
        except ImportError:
            pytest.skip("chromadb not installed")

        from core.rag.vectorstore import SentinelVectorStore
        vs = SentinelVectorStore(
            collection_name=f"test_{uuid.uuid4().hex[:8]}",
            persist_directory=str(tmp_path / "chroma"),
        )
        yield vs
        vs.reset()

    def test_vectorstore_repr(self, vectorstore):
        r = repr(vectorstore)
        assert "SentinelVectorStore" in r
        assert "persistent" in r

    def test_empty_count(self, vectorstore):
        """New collection should have 0 documents."""
        assert vectorstore.count == 0

    def test_add_and_count(self, vectorstore):
        """Should correctly count documents after adding."""
        import numpy as np
        texts = ["document one", "document two", "document three"]
        embeddings = [np.random.randn(384).tolist() for _ in texts]
        ids = vectorstore.add(texts=texts, embeddings=embeddings)
        assert len(ids) == 3
        assert vectorstore.count == 3

    def test_query_returns_results(self, vectorstore):
        """Should return results after adding documents."""
        import numpy as np
        np.random.seed(42)
        texts = ["pump maintenance schedule", "safety protocols", "AI model selection"]
        embeddings = [np.random.randn(384).tolist() for _ in texts]
        vectorstore.add(texts=texts, embeddings=embeddings)

        query_emb = np.random.randn(384).tolist()
        results = vectorstore.query(query_embedding=query_emb, top_k=2)
        assert len(results) == 2
        for r in results:
            assert "id" in r
            assert "text" in r
            assert "score" in r
            assert "metadata" in r

    def test_query_empty_store(self, vectorstore):
        """Should return [] when querying empty store."""
        import numpy as np
        result = vectorstore.query(np.random.randn(384).tolist())
        assert result == []

    def test_add_with_metadata(self, vectorstore):
        """Metadata should be stored and returned."""
        import numpy as np
        vectorstore.add(
            texts=["documented text with metadata"],
            embeddings=[np.random.randn(384).tolist()],
            metadatas=[{"source": "manual.pdf", "page": 3}],
        )
        results = vectorstore.query(np.random.randn(384).tolist(), top_k=1)
        if results:
            assert results[0]["metadata"].get("source") == "manual.pdf"
            assert results[0]["metadata"].get("page") == 3

    def test_delete_documents(self, vectorstore):
        """Should correctly delete documents by ID."""
        import numpy as np
        ids = vectorstore.add(
            texts=["to delete"],
            embeddings=[np.random.randn(384).tolist()],
        )
        assert vectorstore.count == 1
        vectorstore.delete(ids)
        assert vectorstore.count == 0

    def test_reset_clears_all(self, vectorstore):
        """reset() should clear all documents."""
        import numpy as np
        vectorstore.add(
            texts=["a", "b", "c"],
            embeddings=[np.random.randn(384).tolist() for _ in range(3)],
        )
        assert vectorstore.count == 3
        vectorstore.reset()
        assert vectorstore.count == 0

    def test_score_in_valid_range(self, vectorstore):
        """Cosine similarity scores should be in [0, 1] range."""
        import numpy as np
        texts = ["cat", "dog", "car"]
        embeddings = [np.random.randn(384).tolist() for _ in texts]
        vectorstore.add(texts=texts, embeddings=embeddings)
        results = vectorstore.query(np.random.randn(384).tolist(), top_k=3)
        for r in results:
            assert 0.0 <= r["score"] <= 1.0


# =========================================================================== #
# RETRIEVER TESTS (mocked embedder + vectorstore)
# =========================================================================== #

class TestSentinelRetriever:
    @pytest.fixture
    def mock_retriever(self, tmp_path):
        """Build a SentinelRetriever with mocked embedder and real ChromaDB."""
        try:
            import chromadb  # noqa: F401
        except ImportError:
            pytest.skip("chromadb not installed")

        import numpy as np
        from core.rag.retriever import SentinelRetriever

        retriever = SentinelRetriever(
            collection_name=f"test_{uuid.uuid4().hex[:8]}",
            persist_directory=str(tmp_path / "chroma"),
        )

        # Mock the embedder to return deterministic vectors
        mock_embedder = MagicMock()

        def fake_embed(text):
            if isinstance(text, str):
                vec = np.random.randn(384).astype(np.float32)
                return vec / np.linalg.norm(vec)
            else:
                arrs = np.random.randn(len(text), 384).astype(np.float32)
                norms = np.linalg.norm(arrs, axis=1, keepdims=True)
                return arrs / norms

        def fake_embed_batch(texts):
            arrs = np.random.randn(len(texts), 384).astype(np.float32)
            norms = np.linalg.norm(arrs, axis=1, keepdims=True)
            return (arrs / norms).tolist()

        mock_embedder.embed.side_effect = fake_embed
        mock_embedder.embed_batch.side_effect = fake_embed_batch
        mock_embedder.model_name = "mock-model"
        retriever.embedder = mock_embedder

        yield retriever
        retriever.reset()

    def test_retriever_repr(self, mock_retriever):
        r = repr(mock_retriever)
        assert "SentinelRetriever" in r

    def test_ingest_file(self, mock_retriever, sample_text_file):
        """ingest_file should add chunks to the vector store."""
        ids = mock_retriever.ingest_file(sample_text_file)
        assert len(ids) > 0
        assert mock_retriever.document_count > 0

    def test_ingest_directory(self, mock_retriever, sample_docs_dir):
        """ingest_directory should load and store all docs."""
        ids = mock_retriever.ingest_directory(sample_docs_dir)
        assert len(ids) > 0
        assert mock_retriever.document_count == len(ids)

    def test_retrieve_returns_docs(self, mock_retriever, sample_text_file):
        """retrieve() should return a list of ranked docs."""
        mock_retriever.ingest_file(sample_text_file)
        results = mock_retriever.retrieve("LLaVA model inference")
        assert isinstance(results, list)
        for r in results:
            assert "text" in r
            assert "score" in r
            assert "metadata" in r

    def test_retrieve_empty_store(self, mock_retriever):
        """retrieve() on empty store should return empty list."""
        results = mock_retriever.retrieve("any query")
        assert results == []

    def test_retrieve_as_context_string(self, mock_retriever, sample_text_file):
        """retrieve_as_context() should return a formatted string."""
        mock_retriever.ingest_file(sample_text_file)
        context = mock_retriever.retrieve_as_context("Ollama")
        assert context is not None
        assert isinstance(context, str)
        assert len(context) > 10

    def test_retrieve_as_context_empty_store(self, mock_retriever):
        """retrieve_as_context() on empty store should return None."""
        result = mock_retriever.retrieve_as_context("anything")
        assert result is None

    def test_top_k_respected(self, mock_retriever, sample_docs_dir):
        """retrieve() should respect top_k limit."""
        mock_retriever.ingest_directory(sample_docs_dir)
        results = mock_retriever.retrieve("maintenance", top_k=2)
        assert len(results) <= 2

    def test_keyword_rerank(self):
        """_keyword_rerank should boost docs with query terms."""
        from core.rag.retriever import SentinelRetriever
        docs = [
            {"text": "pump maintenance schedule and procedures", "score": 0.7},
            {"text": "unrelated topic about weather", "score": 0.75},
        ]
        reranked = SentinelRetriever._keyword_rerank("pump maintenance", docs)
        # The pump+maintenance doc should score at least as high as weather doc
        pump_doc = next(d for d in reranked if "pump" in d["text"])
        weather_doc = next(d for d in reranked if "weather" in d["text"])
        assert pump_doc["score"] >= weather_doc["score"]

    def test_reset_clears_store(self, mock_retriever, sample_text_file):
        """reset() should empty the vector store."""
        mock_retriever.ingest_file(sample_text_file)
        assert mock_retriever.document_count > 0
        mock_retriever.reset()
        assert mock_retriever.document_count == 0


# =========================================================================== #
# PIPELINE + RAG INTEGRATION TESTS
# =========================================================================== #

class TestPipelineWithRAG:
    @pytest.fixture
    def pipeline_with_mock_rag(self):
        """Pipeline with mocked vision, LLM, RAG retriever, and monitoring."""
        from unittest.mock import patch, MagicMock
        from core.vision.base import VisionResult
        from core.llm.ollama_client import LLMResponse

        with patch("core.pipeline.get_vision_model") as mock_vision_factory, \
             patch("core.pipeline.OllamaLLMClient") as mock_llm_cls, \
             patch.object(
                 __import__("core.pipeline", fromlist=["SentinelPipeline"]).SentinelPipeline,
                 "_init_rag", return_value=None
             ), \
             patch.object(
                 __import__("core.pipeline", fromlist=["SentinelPipeline"]).SentinelPipeline,
                 "_init_monitoring", return_value=None
             ):

            mock_vision = MagicMock()
            mock_vision.name = "MockVision"
            mock_vision._loaded = False
            mock_vision.analyze.return_value = VisionResult(
                model_name="MockVision",
                description="A machine part with visible wear marks.",
                latency_ms=50.0,
                memory_mb=10.0,
            )
            mock_vision_factory.return_value = mock_vision

            mock_llm = MagicMock()
            mock_llm.model = "llama3.1:8b"
            mock_llm.is_available.return_value = True
            mock_llm.generate.return_value = LLMResponse(
                model="llama3.1:8b",
                content="Based on the visible wear marks, pump maintenance is recommended.",
                prompt_tokens=60,
                completion_tokens=20,
                total_tokens=80,
                latency_ms=200.0,
            )
            mock_llm_cls.return_value = mock_llm

            from core.pipeline import SentinelPipeline
            p = SentinelPipeline(vision_model="llava", device="cpu", enable_rag=True)
            p.vision = mock_vision
            p.llm = mock_llm

            # Explicitly set monitoring attrs to None (no-op path in pipeline)
            p._tracer = None
            p._metrics = None

            # Mock the RAG retriever
            mock_retriever = MagicMock()
            mock_retriever.retrieve_as_context.return_value = (
                "[Source: maintenance.md | Score: 0.82]\n"
                "Pumps should be inspected every 2000 hours."
            )
            mock_retriever.document_count = 15
            p._retriever = mock_retriever

            yield p

    def test_pipeline_calls_rag(self, pipeline_with_mock_rag, tmp_path):
        """Pipeline should call RAG retriever when enabled."""
        from PIL import Image
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (64, 64), color=(100, 100, 100)).save(img_path)

        result = pipeline_with_mock_rag.run_image(img_path, "What maintenance is needed?")
        pipeline_with_mock_rag._retriever.retrieve_as_context.assert_called_once()

    def test_rag_context_in_result(self, pipeline_with_mock_rag, tmp_path):
        """Pipeline result should include RAG context."""
        from PIL import Image
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (64, 64)).save(img_path)

        result = pipeline_with_mock_rag.run_image(img_path, "Check for damage.")
        assert result.rag_context is not None
        assert "maintenance" in result.rag_context.lower()

    def test_pipeline_rag_chunks_count(self, pipeline_with_mock_rag, tmp_path):
        """rag_chunks_retrieved should reflect the retrieved context."""
        from PIL import Image
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (64, 64)).save(img_path)

        result = pipeline_with_mock_rag.run_image(img_path)
        assert result.rag_chunks_retrieved >= 0

    def test_health_check_includes_rag(self, pipeline_with_mock_rag):
        """health_check() should include rag_enabled and rag_docs."""
        health = pipeline_with_mock_rag.health_check()
        assert "rag_enabled" in health
        assert health["rag_enabled"] is True
        assert "rag_docs" in health
        assert health["rag_docs"] == 15

    def test_build_prompt_with_rag_context(self):
        """_build_llm_prompt should include RAG section when context provided."""
        from core.pipeline import SentinelPipeline
        from core.vision.base import VisionResult
        vr = VisionResult(model_name="Test", description="A pump with worn seals.")
        rag_ctx = "[Source: manual.pdf]\nReplace seals every 2000 hours."

        prompt = SentinelPipeline._build_llm_prompt("What should I do?", vr, rag_ctx)
        assert "RELEVANT CONTEXT FROM KNOWLEDGE BASE" in prompt
        assert "Replace seals" in prompt

    def test_build_prompt_without_rag_context(self):
        """_build_llm_prompt without RAG should not include knowledge base section."""
        from core.pipeline import SentinelPipeline
        from core.vision.base import VisionResult
        vr = VisionResult(model_name="Test", description="A robot arm.")
        prompt = SentinelPipeline._build_llm_prompt("Describe.", vr, None)
        assert "RELEVANT CONTEXT FROM KNOWLEDGE BASE" not in prompt
