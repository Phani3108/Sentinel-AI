"""
Sentinel AI — FastAPI Server
Provides REST + WebSocket endpoints for the multimodal pipeline.
"""
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from core.config import get_settings
from core.pipeline import SentinelPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# App State
# ------------------------------------------------------------------ #
_pipeline: Optional[SentinelPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipeline on startup, clean up on shutdown."""
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
    version="0.1.0",
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


class HealthResponse(BaseModel):
    status: str
    vision_model: str
    llm_model: str
    llm_available: bool
    device: str


# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check — verify pipeline components are running."""
    health = _pipeline.health_check()
    return HealthResponse(
        status="ok",
        vision_model=health["vision_model"],
        llm_model=health["llm_model"],
        llm_available=health["llm_available"],
        device=health["device"],
    )


@app.post("/analyze/image", response_model=AnalyzeResponse, tags=["Inference"])
async def analyze_image(
    file: UploadFile = File(..., description="Image file to analyze"),
    prompt: str = Form(default="What do you see in this image?"),
    vision_prompt: Optional[str] = Form(default=None),
):
    """
    Analyze a single image through the full pipeline:
    image → vision model → LLM reasoning → response
    """
    # Save uploaded file to temp location
    suffix = Path(file.filename).suffix if file.filename else ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        result = _pipeline.run_image(
            tmp_path,
            prompt=prompt,
            vision_prompt=vision_prompt,
        )
        return AnalyzeResponse(
            vision_model=result.vision_model,
            vision_description=result.vision_description,
            final_answer=result.final_answer,
            total_latency_ms=result.total_latency_ms,
            llm_tokens_used=result.llm_tokens_used,
        )
    finally:
        os.unlink(tmp_path)


@app.post("/analyze/image/stream", tags=["Inference"])
async def analyze_image_stream(
    file: UploadFile = File(...),
    prompt: str = Form(default="What do you see in this image?"),
):
    """
    Stream vision + LLM response tokens for an image.
    Returns Server-Sent Events (SSE) format.
    """

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
            os.unlink(tmp_path)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.websocket("/ws/analyze", name="websocket_analyze")
async def websocket_analyze(websocket: WebSocket):
    """
    WebSocket endpoint for real-time image analysis.
    
    Protocol:
        Client → Server: JSON { "prompt": str, "image_b64": str }
        Server → Client: streaming tokens, then JSON summary
    """
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

            # Decode and save temp image
            import base64
            img_bytes = base64.b64decode(image_b64)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(img_bytes)
                tmp_path = Path(tmp.name)

            try:
                # Stream tokens back
                for token in _pipeline.stream_image(tmp_path, prompt=prompt):
                    await websocket.send_text(token)
                await websocket.send_json({"done": True})
            except Exception as e:
                await websocket.send_json({"error": str(e)})
            finally:
                os.unlink(tmp_path)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")


@app.get("/models", tags=["System"])
async def list_models():
    """List all available vision models in the registry."""
    from core.vision import _REGISTRY
    return {
        "vision_models": list(_REGISTRY.keys()),
        "current_vision_model": _pipeline.vision.name if _pipeline else None,
        "current_llm_model": _pipeline.llm.model if _pipeline else None,
    }


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
