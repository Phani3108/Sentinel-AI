import base64
import asyncio
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

live_router = APIRouter(prefix="/live", tags=["Live Analytics"])

def _process_frame_sync(pipeline, image_bytes: bytes, tripwire: str) -> dict:
    """Run the heavy synchronous LLM pipeline in an offloaded thread."""
    # Force the LLM to yield a strict JSON threat profile
    prompt = f"TRIPWIRE: {tripwire}\nDetermine if this is breached."
    
    # Bypass streams to get the final JSON string
    result = pipeline.run_image(image_bytes, prompt, use_agent=False)
    
    # Extremely basic heuristic parsing since local models might hallucinate strict JSON
    ans = result.final_answer.lower()
    is_triggered = False
    if "'triggered': true" in ans or '"triggered": true' in ans or "yes" in ans or ("true" in ans and "false" not in ans):
        is_triggered = True
        
    return {
        "triggered": is_triggered,
        "reason": result.final_answer,
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
            
            # Retrieve global singleton pipeline
            pipeline = websocket.app.state.pipeline

            # Offload heavy synchronous ML inference to the AsyncIO threadpool
            eval_result = await asyncio.to_thread(_process_frame_sync, pipeline, image_bytes, tripwire)
            
            await websocket.send_json({
                "status": "success",
                "triggered": eval_result["triggered"],
                "inference": eval_result,
                "latency_ms": 0 # TODO: wire up latency
            })
            
    except WebSocketDisconnect:
        logger.info("Live Monitoring Client Disconnected.")
    except Exception as e:
        logger.error(f"Live websocket unhandled fault: {e}")


@live_router.websocket("/sensor")
async def sensor_stream(websocket: WebSocket):
    """
    Dedicated enterprise ingest port for disconnected Edge Nodes (e.g. Raspberry Pis).
    Enforces JWT/Node-Token authentication to prevent unauthorized video uploads.
    """
    from core.fleet import fleet_manager
    
    node_token = websocket.headers.get("X-Node-Token")
    if node_token != "edge-device-secret-123":
        logger.warning(f"Unauthorized edge node connection attempt. Token: {node_token}")
        await websocket.close(code=1008)
        return
        
    await websocket.accept()
    node_id = fleet_manager.register_node()
    logger.info(f"Physical EDGE NODE uplink authenticated. Fleet ID: {node_id}")
    
    tripwire = "Analyze environment for generic threats."
    
    try:
        while True:
            data_str = await websocket.receive_text()
            fleet_manager.ping_node(node_id)
            try:
                msg = json.loads(data_str)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Strict JSON required."})
                continue
                
            if msg.get("type") == "init":
                tripwire = msg.get("tripwire", tripwire)
                logger.info(f"Edge Node initialized tripwire: {tripwire}")
                continue
                
            if msg.get("type") == "frame":
                frame_b64 = msg.get("frame", "")
                
                try:
                    image_bytes = base64.b64decode(frame_b64)
                except Exception as e:
                    await websocket.send_json({"error": f"Invalid frame encoding: {str(e)}"})
                    continue
                
                # Retrieve global singleton
                pipeline = websocket.app.state.pipeline

                # Offload to AI Executor
                eval_result = await asyncio.to_thread(_process_frame_sync, pipeline, image_bytes, tripwire)
                
                await websocket.send_json({
                    "status": "success",
                    "inference": eval_result
                })
                
    except WebSocketDisconnect:
        logger.info("Edge Node Uplink severed.")
    except Exception as e:
        logger.error(f"Edge Node unhandled fault: {e}")
    finally:
        from core.fleet import fleet_manager
        fleet_manager.disconnect_node(node_id)
