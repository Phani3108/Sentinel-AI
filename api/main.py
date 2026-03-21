"""
Sentinel AI — FastAPI Server (v0.4.0)
Phase 7: Enterprise Security & Governance
Provides REST + WebSocket endpoints with strict API Key Auth, RBAC, PII Masking, Rate Limiting, and Audit Logging.
"""
import base64
import hashlib
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.staticfiles import StaticFiles

from core.config import get_settings
from core.pipeline import SentinelPipeline

# Phase 8 Caching & Async Jobs
from core.cache import get_cache
from api.jobs import router as jobs_router

# Phase 9 Feedback Loop
from api.feedback import router as feedback_router

# Phase 12 Live Engine
from api.live import live_router

# Phase 14 Swarm Intelligence
from api.swarm import swarm_router

# Phase 19 Advanced Threat Hunting
from api.hunting import hunting_router

# Phase 7 Security Modules
from api.security.auth import get_current_user, UserAccount, UserRole
from api.security.rbac import require_role
from core.audit.logger import get_audit_logger
from core.security.masking import get_pii_masker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# App State
# ------------------------------------------------------------------ #
_pipeline: Optional[SentinelPipeline] = None
_retriever = None

limiter = Limiter(key_func=get_remote_address)

def hash_file(filepath: Path) -> str:
    """Helper for audit logs to track the image identity without storing the image."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

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
    # Prime singletons
    get_audit_logger()
    get_cache()
    # Inject into global context for websocket routers avoiding circular imports
    app.state.pipeline = _pipeline
    yield
    logger.info("Shutting down Sentinel AI API...")


# ------------------------------------------------------------------ #
# FastAPI App
# ------------------------------------------------------------------ #
app = FastAPI(
    title="Sentinel AI Enterprise Engine",
    description="Multimodal Analytics, ReAct Orchestration, & Live Threat Telemetry",
    version="10.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(jobs_router)
app.include_router(feedback_router)
app.include_router(live_router)
app.include_router(swarm_router)
app.include_router(hunting_router)

# Mount static metrics dashboard
app.mount("/metrics-dashboard", StaticFiles(directory="monitoring/dashboards"), name="metrics-ui")

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

@app.post("/system/mock/slack", tags=["System"])
async def mock_slack_webhook(request: Request):
    """
    Simulated Enterprise Slack web hook. 
    Verifies that the Action Agent can physically craft and POST HTTP JSON payloads to remote servers.
    """
    payload = await request.json()
    logger.info(f"MOCK SLACK RECEIVED WEBHOCK: {payload.get('text')}")
    # Echos back the receipt for the Action Node to verify success
    return {"status": "received_by_slack_mock", "message_length": len(payload.get("text", ""))}

@app.get("/system/audit/verify", tags=["System"])
def verify_audit_ledger():
    """Mathematically recurses the SQL database to prove logs have not been retroactively tapered with."""
    from core.security.ledger import ImmutableLedger
    ledger = ImmutableLedger()
    return ledger.verify_chain()

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check — intentionally left public so Load Balancers can ping it."""
    h = _pipeline.health_check()
    return HealthResponse(
        status="ok",
        vision_model=h["vision_model"],
        llm_model=h["llm_model"],
        llm_available=h["llm_available"],
        device=h["device"],
    )


@app.get("/models", tags=["System"])
async def list_models(user: UserAccount = Depends(get_current_user)):
    """List all available vision models. Requires authentication."""
    from core.vision import _REGISTRY
    return {
        "vision_models": list(_REGISTRY.keys()),
        "current_vision_model": _pipeline.vision.name if _pipeline else None,
        "current_llm_model": _pipeline.llm.model if _pipeline else None,
    }


@app.get("/metrics", tags=["System"])
async def prometheus_metrics():
    """Expose Prometheus metrics. Usually scraped by internal Prometheus server (safe public)."""
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
@limiter.limit("10/minute")
async def analyze_image(
    request: Request,
    file: UploadFile = File(...),
    prompt: str = Form(default="What do you see in this image?"),
    vision_prompt: Optional[str] = Form(default=None),
    user: UserAccount = Depends(get_current_user)
):
    """Analyze a single image. Authenticated, Rate Limited, and Audited."""
    # 1. PII Masking
    redacted_prompt, mask_stats = get_pii_masker().redact(prompt)
    
    suffix = Path(file.filename).suffix if file.filename else ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)
        img_hash = hash_file(tmp_path)
    
    try:
        # Check cache early to bypass heavy GPU
        cached_result = get_cache().get_inference(img_hash, redacted_prompt)
        if cached_result:
            return AnalyzeResponse(**cached_result)

        # 2. Execution
        result = _pipeline.run_image(tmp_path, prompt=redacted_prompt, vision_prompt=vision_prompt)
        
        # 3. Audit Logging
        img_hash = hash_file(tmp_path)
        get_audit_logger().log_request(
            username=user.username,
            endpoint="/analyze/image",
            prompt=redacted_prompt,
            image_hash=img_hash,
            route="local",
            status_code=200,
            response_meta={"latency": result.total_latency_ms, "pii_masked": mask_stats, "cache": "MISS"}
        )
        
        response_dict = {
            "vision_model": result.vision_model,
            "vision_description": result.vision_description,
            "final_answer": result.final_answer,
            "total_latency_ms": result.total_latency_ms,
            "llm_tokens_used": result.llm_tokens_used,
            "rag_context": getattr(result, "rag_context", None),
            "rag_chunks_retrieved": getattr(result, "rag_chunks_retrieved", 0),
        }
        
        # Save exact-match result back to cache for 24h
        get_cache().set_inference(img_hash, redacted_prompt, response_dict)

        return AnalyzeResponse(**response_dict)
    except Exception as e:
        get_audit_logger().log_request(
            username=user.username, endpoint="/analyze/image", status_code=500, prompt=redacted_prompt
        )
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.post("/analyze/image/stream", tags=["Inference"])
@limiter.limit("20/minute")
async def analyze_image_stream(
    request: Request,
    file: UploadFile = File(...),
    prompt: str = Form(default="What do you see in this image?"),
    user: UserAccount = Depends(get_current_user)
):
    """Stream vision + LLM response tokens via Server-Sent Events."""
    redacted_prompt, _ = get_pii_masker().redact(prompt)
    
    suffix = Path(file.filename).suffix if file.filename else ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    async def generate():
        try:
            get_audit_logger().log_request(
                username=user.username, endpoint="/analyze/image/stream", prompt=redacted_prompt, status_code=200
            )
            for token in _pipeline.stream_image(tmp_path, prompt=redacted_prompt):
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
@limiter.limit("2/minute")
async def analyze_video(
    request: Request,
    file: UploadFile = File(..., description="Video file to analyze"),
    prompt: str = Form(default="Describe what is happening in this video."),
    fps: float = Form(default=0.5),
    max_frames: int = Form(default=15),
    use_keyframes: bool = Form(default=False),
    user: UserAccount = Depends(get_current_user)
):
    """Analyse a video file."""
    redacted_prompt, mask_stats = get_pii_masker().redact(prompt)
    
    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)
        
    try:
        result = _pipeline.run_video(tmp_path, prompt=redacted_prompt, fps=fps, max_frames=max_frames)
        frame_summaries = []
        for fs in getattr(result, "frame_summaries", []):
            frame_summaries.append({
                "timestamp": getattr(fs, "timestamp_s", 0),
                "description": getattr(fs, "vision_description", ""),
                "latency_ms": getattr(fs, "latency_ms", 0),
            })
            
        get_audit_logger().log_request(
            username=user.username, endpoint="/analyze/video", prompt=redacted_prompt, status_code=200,
            response_meta={"total_frames": len(frame_summaries), "latency": result.total_latency_ms}
        )
        return {
            "narration": result.final_answer,
            "total_frames": len(frame_summaries),
            "frame_summaries": frame_summaries,
            "total_latency_ms": result.total_latency_ms,
        }
    except Exception as e:
        get_audit_logger().log_request(username=user.username, endpoint="/analyze/video", status_code=500)
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
@limiter.limit("5/minute")
async def rag_ingest(
    request: Request,
    file: UploadFile = File(...),
    chunk_size: int = Form(default=500),
    overlap: int = Form(default=50),
    user: UserAccount = Depends(require_role(UserRole.ADMIN))
):
    """Ingest a document into the vector store. Requires ADMIN role."""
    try:
        retriever = _get_retriever()
        suffix = Path(file.filename).suffix if file.filename else ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)
        try:
            ids = retriever.ingest_file(tmp_path, chunk_size=chunk_size, overlap=overlap)
            get_audit_logger().log_request(username=user.username, endpoint="/rag/ingest", response_meta={"chunks": len(ids)})
            return {"chunks_added": len(ids), "filename": file.filename}
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/rag/query", tags=["RAG"])
@limiter.limit("20/minute")
async def rag_query(
    request: Request,
    query_req: RAGQueryRequest,
    user: UserAccount = Depends(get_current_user)
):
    """Query the vector store. Uses Masking to prevent pulling PII."""
    redacted_query, _ = get_pii_masker().redact(query_req.query)
    try:
        retriever = _get_retriever()
        results = retriever.retrieve(redacted_query, top_k=query_req.top_k)
        get_audit_logger().log_request(username=user.username, endpoint="/rag/query", prompt=redacted_query)
        return {"results": results, "query": redacted_query}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/rag/stats", tags=["RAG"])
async def rag_stats(user: UserAccount = Depends(get_current_user)):
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
@limiter.limit("1/minute")
async def rag_reset(
    request: Request,
    user: UserAccount = Depends(require_role(UserRole.ADMIN))
):
    """Clear all documents from the vector store. Requires ADMIN role."""
    try:
        retriever = _get_retriever()
        retriever.reset()
        get_audit_logger().log_request(username=user.username, endpoint="/rag/reset", response_meta={"status": "cleared"})
        return {"status": "ok", "message": "Collection reset"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ------------------------------------------------------------------ #
# Hybrid Router Endpoint
# ------------------------------------------------------------------ #

@app.post("/route/image", tags=["Router"])
@limiter.limit("10/minute")
async def route_image(
    request: Request,
    file: UploadFile = File(...),
    prompt: str = Form(default="What do you see in this image?"),
    strict_mode: bool = Form(default=False),
    user: UserAccount = Depends(get_current_user)
):
    """Route an image through the hybrid router. Enforces Security first."""
    # Mask prompts going to the cloud
    redacted_prompt, mask_stats = get_pii_masker().redact(prompt)
    
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
        result = router.route(str(tmp_path), redacted_prompt)
        
        get_audit_logger().log_request(
            username=user.username,
            endpoint="/route/image",
            prompt=redacted_prompt,
            image_hash=hash_file(tmp_path),
            route=result.route,
            status_code=200,
            response_meta={"cost": result.estimated_cost_usd, "classification": result.classification_label}
        )
        
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
        get_audit_logger().log_request(username=user.username, endpoint="/route/image", status_code=500)
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ------------------------------------------------------------------ #
# WebSocket (WebSockets do not easily accept standard Authorization HTTP headers in browser JS, 
# so we pass query params or accept insecurely for internal dashboard only)
# ------------------------------------------------------------------ #

@app.websocket("/ws/analyze")
async def websocket_analyze(websocket: WebSocket):
    """WebSocket endpoint for real-time image analysis. 
    NOTE: Secure token parsing inside WS is complex; left simple for now."""
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    try:
        while True:
            data = await websocket.receive_json()
            prompt = data.get("prompt", "Describe this image.")
            # Redact prompt before sending
            redacted_prompt, _ = get_pii_masker().redact(prompt)
            
            image_b64 = data.get("image_b64")
            if not image_b64:
                await websocket.send_json({"error": "image_b64 required"})
                continue
            img_bytes = base64.b64decode(image_b64)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(img_bytes)
                tmp_path = Path(tmp.name)
            try:
                get_audit_logger().log_request(
                    username="websocket_client", endpoint="/ws/analyze", prompt=redacted_prompt, status_code=200
                )
                for token in _pipeline.stream_image(tmp_path, prompt=redacted_prompt):
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
