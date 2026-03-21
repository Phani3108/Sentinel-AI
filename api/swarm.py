import json
import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import StreamingResponse

from core.swarm.graph import compile_swarm

logger = logging.getLogger(__name__)
swarm_router = APIRouter(prefix="/swarm", tags=["Swarm Intelligence"])

@swarm_router.post("/stream")
async def swarm_stream(
    file: UploadFile = File(...),
    tripwire: str = Form(default="Analyze environment for threats.")
):
    """
    Initiates the Multi-Agent Swarm against a target visual payload.
    Streams back LangGraph execution state via Server-Sent Events.
    """
    suffix = Path(file.filename).suffix if file.filename else ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    # In a production app, compiling the graph is typically done once at startup.
    # For Sandbox safety, we compute it on request.
    swarm = compile_swarm()

    async def generate():
        try:
            initial_state = {
                "tripwire": tripwire,
                "image_path": tmp_path,
                "messages": [],
                "vision_context": "",
                "intel_context": "",
                "decision": "",
                "threat_level": ""
            }
            
            # The LangGraph stream yields the dictionary output computed by each Node 
            for output in swarm.stream(initial_state):
                for node_name, state_update in output.items():
                    messages = state_update.get("messages", [])
                    if messages:
                        # Extract the physical text message and strip newline breaks to satisfy SSE protocol
                        raw_content = messages[-1].content
                        safe_content = raw_content.replace('\n', '  ')
                        
                        payload = json.dumps({"node": node_name, "message": safe_content})
                        logger.info(f"SSE Yield -> {node_name}")
                        yield f"data: {payload}\n\n"
                    
            yield "data: {\"node\": \"system\", \"message\": \"[DONE]\"}\n\n"
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return StreamingResponse(generate(), media_type="text/event-stream")
