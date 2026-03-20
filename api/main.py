"""
Sentinel AI — FastAPI Server (v0.3.0)
Provides REST + WebSocket endpoints for the multimodal pipeline.
Phase 4 additions: RAG endpoints, video endpoint, hybrid router endpoint.
"""
import base64
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from core.config import get_settings
from core.pipeline import SentinelPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# App State
# ------------------------------------------------------------------ #
_pipeline: Optional[SentinelPipeline] = None
_retriever = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    settings = get_settings()
    _pipeline = SentinelPipeline(
        vision_model=settings.default_vision_model,
        llm_model=settings.ollama_llm_model,
        device=settings.device,
        enable_rag=False,
    )
    logger.info(f"✅ Pipeline initialized: {_pipeline}")
    yield
    logger.info("Shutting down Sentinel AI API...")


# ------------------------------------------------------------------ #
# FastAPI App
# ------------------------------------------------------------------ #
app = FastAPI(
    title="Sentinel AI",
    description="Private Multimodal AI Stack — On-Prem Vision + LLM",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ #
# Request / Response Models
# ------------------------------------------------------------------ #
class AnalyzeResponse(BaseModel):
    vision_model: str
    vision_description: str
    final_answer: str
    total_latency_ms: float
    llm_tokens_used: int
    rag_context: Optional[str] = None
    rag_chunks_retrieved: int = 0


class HealthResponse(BaseModel):
    status: str
    vision_model: str
    llm_model: str
    llm_available: bool
    device: str


class RAGQueryRequest(BaseModel):
    query: str
    top_k: int = 3


# ------------------------------------------------------------------ #
# System Endpoints
# ------------------------------------------------------------------ #

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check — verify pipeline components are running."""
    h = _pipeline.health_check()
    return HealthResponse(
        status="ok",
        vision_model=h["vision_model"],
        llm_model=h["llm_model"],
        llm_available=h["llm_available"],
        device=h["device"],
    )


@app.get("/models", tags=["System"])
async def list_models():
    """List all available vision models in the registry."""
    from core.vision import _REGISTRY
    return {
        "vision_models": list(_REGISTRY.keys()),
        "current_vision_model": _pipeline.vision.name if _pipeline else None,
        "current_llm_model": _pipeline.llm.model if _pipeline else None,
    }


@app.get("/metrics", tags=["System"])
async def prometheus_metrics():
    """Expose Prometheus metrics in text format."""
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return PlainTextResponse(
            content=generate_latest().decode("utf-8"),
            media_type=CONTENT_TYPE_LATEST,
        )
    except ImportError:
        return PlainTextResponse(content="# prometheus_client not installed\n", media_type="text/plain")


# ------------------------------------------------------------------ #
# Image Inference Endpoints
# ------------------------------------------------------------------ #

@app.post("/analyze/image", response_model=AnalyzeResponse, tags=["Inference"])
async def analyze_image(
    file: UploadFile = File(..., description="Image file to analyze"),
    prompt: str = Form(default="What do you see in this image?"),
    vision_prompt: Optional[str] = Form(default=None),
):
    """Analyze a single image: image → vision model → LLM → response."""
    suffix = Path(file.filename).suffix if file.filename else ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        result = _pipeline.run_image(tmp_path, prompt=prompt, vision_prompt=vision_prompt)
        return AnalyzeResponse(
            vision_model=result.vision_model,
            vision_description=result.vision_description,
            final_answer=result.final_answer,
            total_latency_ms=result.total_latency_ms,
            llm_tokens_used=result.llm_tokens_used,
            rag_context=getattr(result, "rag_context", None),
            rag_chunks_retrieved=getattr(result, "rag_chunks_retrieved", 0),
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.post("/analyze/image/stream", tags=["Inference"])
async def analyze_image_stream(
    file: UploadFile = File(...),
    prompt: str = Form(default="What do you see in this image?"),
):
    """Stream vision + LLM response tokens via Server-Sent Events."""
    suffix = Path(file.filename).suffix if file.filename else ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    async def generate():
        try:
            for token in _pipeline.stream_image(tmp_path, prompt=prompt):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return StreamingResponse(generate(), media_type="text/event-stream")


# ------------------------------------------------------------------ #
# Video Inference Endpoints
# ------------------------------------------------------------------ #

@app.post("/analyze/video", tags=["Inference"])
async def analyze_video(
    file: UploadFile = File(..., description="Video file to analyze"),
    prompt: str = Form(default="Describe what is happening in this video."),
    fps: float = Form(default=0.5),
    max_frames: int = Form(default=15),
    use_keyframes: bool = Form(default=False),
):
    """Analyse a video file: extract frames → per-frame vision → LLM narration."""
    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        result = _pipeline.run_video(tmp_path, prompt=prompt, fps=fps, max_frames=max_frames)
        frame_summaries = []
        for fs in getattr(result, "frame_summaries", []):
            frame_summaries.append({
                "timestamp": getattr(fs, "timestamp_s", 0),
                "description": getattr(fs, "vision_description", ""),
                "latency_ms": getattr(fs, "latency_ms", 0),
            })
        return {
            "narration": result.final_answer,
            "total_frames": len(frame_summaries),
            "frame_summaries": frame_summaries,
            "total_latency_ms": result.total_latency_ms,
        }
    except Exception as e:
        logger.error(f"Video analysis failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ------------------------------------------------------------------ #
# RAG Endpoints
# ------------------------------------------------------------------ #

@app.post("/rag/ingest", tags=["RAG"])
async def rag_ingest(
    file: UploadFile = File(...),
    chunk_size: int = Form(default=500),
    overlap: int = Form(default=50),
):
    """Ingest a document into the vector store."""
    try:
        retriever = _get_retriever()
        suffix = Path(file.filename).suffix if file.filename else ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)
        try:
            ids = retriever.ingest_file(tmp_path, chunk_size=chunk_size, overlap=overlap)
            return {"chunks_added": len(ids), "filename": file.filename}
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/rag/ingest/samples", tags=["RAG"])
async def rag_ingest_samples():
    """Ingest the built-in sample knowledge base documents from data/docs/."""
    try:
        retriever = _get_retriever()
        docs_dir = Path("data/docs")
        if not docs_dir.exists():
            return JSONResponse(status_code=404, content={"error": "data/docs not found"})
        ids = retriever.ingest_directory(docs_dir)
        return {"chunks_added": len(ids), "source": str(docs_dir)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/rag/query", tags=["RAG"])
async def rag_query(request: RAGQueryRequest):
    """Query the vector store and return ranked chunks."""
    try:
        retriever = _get_retriever()
        results = retriever.retrieve(request.query, top_k=request.top_k)
        return {"results": results, "query": request.query}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/rag/stats", tags=["RAG"])
async def rag_stats():
    """Return collection statistics."""
    try:
        retriever = _get_retriever()
        return {
            "document_count": retriever.document_count,
            "collection_name": retriever.collection_name,
            "embedding_model": retriever.embedder.model_name,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.delete("/rag/reset", tags=["RAG"])
async def rag_reset():
    """Clear all documents from the vector store."""
    try:
        retriever = _get_retriever()
        retriever.reset()
        return {"status": "ok", "message": "Collection reset"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ------------------------------------------------------------------ #
# Hybrid Router Endpoint
# ------------------------------------------------------------------ #

@app.post("/route/image", tags=["Router"])
async def route_image(
    file: UploadFile = File(...),
    prompt: str = Form(default="What do you see in this image?"),
    strict_mode: bool = Form(default=False),
):
    """Route an image through the hybrid router (local vs cloud based on sensitivity)."""
    suffix = Path(file.filename).suffix if file.filename else ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        from router.hybrid_router import HybridRouter
        from router.cloud_client import CloudClient
        cloud = CloudClient()
        router = HybridRouter(
            local_pipeline=_pipeline,
            cloud_client=cloud if cloud.is_available() else None,
            strict_mode=strict_mode,
        )
        result = router.route(str(tmp_path), prompt)
        return {
            "route": result.route,
            "classification": result.classification_label,
            "classification_score": result.classification_score,
            "classification_reasons": result.classification_reasons,
            "final_answer": result.final_answer,
            "model_used": result.model_used,
            "total_latency_ms": result.total_latency_ms,
            "estimated_cost_usd": result.estimated_cost_usd,
            "success": result.success,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ------------------------------------------------------------------ #
# WebSocket
# ------------------------------------------------------------------ #

@app.websocket("/ws/analyze")
async def websocket_analyze(websocket: WebSocket):
    """WebSocket endpoint for real-time image analysis (base64 image)."""
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    try:
        while True:
            data = await websocket.receive_json()
            prompt = data.get("prompt", "Describe this image.")
            image_b64 = data.get("image_b64")
            if not image_b64:
                await websocket.send_json({"error": "image_b64 required"})
                continue
            img_bytes = base64.b64decode(image_b64)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(img_bytes)
                tmp_path = Path(tmp.name)
            try:
                for token in _pipeline.stream_image(tmp_path, prompt=prompt):
                    await websocket.send_text(token)
                await websocket.send_json({"done": True})
            except Exception as e:
                await websocket.send_json({"error": str(e)})
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _get_retriever():
    """Lazy-initialize the RAG retriever singleton."""
    global _retriever
    if _retriever is None:
        from core.rag import SentinelRetriever
        _retriever = SentinelRetriever()
    return _retriever


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
