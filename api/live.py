import base64
import asyncio
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from core.pipeline import ImagePipeline
from core.llm.ollama_client import OllamaLLMClient

logger = logging.getLogger(__name__)

live_router = APIRouter(prefix="/live", tags=["Live Analytics"])

# Singleton pipeline instances for the websocket worker
_llm = OllamaLLMClient()
_pipeline = ImagePipeline(_llm)

def _process_frame_sync(image_bytes: bytes, tripwire: str) -> dict:
    """Run the heavy synchronous LLM pipeline in an offloaded thread."""
    # Force the LLM to yield a strict JSON threat profile
    system_prompt = (
        "You are an elite automated Sentinel threat detection node. "
        "Analyze the visual description and determine if the strictly defined user TRIPWIRE is breached. "
        "Return EXACTLY this JSON format and nothing else: {'triggered': true/false, 'reason': 'short explanation'}"
    )
    
    prompt = f"TRIPWIRE: {tripwire}\nDetermine if this is breached."
    
    # Bypass streams to get the final JSON string
    result = _pipeline.run_image(image_bytes, prompt, use_agent=False)
    
    # Extremely basic heuristic parsing since local models might hallucinate strict JSON
    ans = result.answer.lower()
    is_triggered = False
    if "'triggered': true" in ans or '"triggered": true' in ans or "true" in ans and "false" not in ans:
        is_triggered = True
        
    return {
        "triggered": is_triggered,
        "reason": result.answer,
        "vision_context": result.vision_description
    }

@live_router.websocket("/stream")
async def live_video_stream(websocket: WebSocket):
    """
    Bi-directional WebSocket for processing live WebRTC frames from the Next.js frontend.
    Yields evaluation JSONs against continuous tripwire rules.
    """
    await websocket.accept()
    logger.info("Live WebSocket stream connected.")
    
    try:
        while True:
            data = await websocket.receive_json()
            frame_b64 = data.get("frame")
            tripwire = data.get("tripwire", "Is there a threat?")
            
            if not frame_b64:
                await websocket.send_json({"error": "No frame provided"})
                continue
                
            # Clean Base64 prefix injected by JS Canvas
            if "," in frame_b64:
                frame_b64 = frame_b64.split(",")[1]
            
            try:
                image_bytes = base64.b64decode(frame_b64)
            except Exception as e:
                await websocket.send_json({"error": f"Invalid frame encoding: {str(e)}"})
                continue

            # Offload heavy synchronous ML inference to the AsyncIO threadpool
            eval_result = await asyncio.to_thread(_process_frame_sync, image_bytes, tripwire)
            
            await websocket.send_json({
                "status": "success",
                "triggered": eval_result["triggered"],
                "analysis": eval_result["reason"],
                "vision": eval_result["vision_context"]
            })
            
    except WebSocketDisconnect:
        logger.info("Live WebSocket disconnected by client.")
    except Exception as e:
        logger.error(f"WebSocket fatal error: {e}")
