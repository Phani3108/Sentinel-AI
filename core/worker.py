"""
Sentinel AI — Celery Async Worker Definitions
Phase 8: High Availability & Scaling

Provides `@app.task` decorators for running long workloads (like full video processing)
in a background queue without timing out FastAPIs HTTP connections.
"""
import os
import logging
from pathlib import Path

try:
    from celery import Celery
    celery_available = True
except ImportError:
    celery_available = False

logger = logging.getLogger(__name__)

# Initialize Celery pointing to Redis as both the message broker and result backend.
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

if celery_available:
    celery_app = Celery(
        'sentinel_tasks',
        broker=REDIS_URL,
        backend=REDIS_URL
    )
    # Configure Celery serialization to allow arbitrary dicts
    celery_app.conf.update(
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
    )
else:
    celery_app = None


# We lazily initialize the pipeline inside the worker processes so we don't 
# load massive models into the master process simply during module import.
_worker_pipeline = None

def get_worker_pipeline():
    global _worker_pipeline
    if _worker_pipeline is None:
        from core.pipeline import SentinelPipeline
        from core.config import get_settings
        settings = get_settings()
        logger.info("Initializing heavy SentinelPipeline inside Celery Worker...")
        _worker_pipeline = SentinelPipeline(
            vision_model=settings.default_vision_model,
            llm_model=settings.ollama_llm_model,
            device=settings.device,
            enable_rag=False
        )
    return _worker_pipeline

if celery_app:
    @celery_app.task(bind=True, name="sentinel.analyze_video")
    def async_analyze_video(self, video_path: str, prompt: str, fps: float, max_frames: int):
        """
        Background task to extract and analyze frames of a video.
        """
        logger.info(f"Starting async video analysis: Job {self.request.id} on {video_path}")
        pipeline = get_worker_pipeline()
        
        path_obj = Path(video_path)
        if not path_obj.exists():
            return {"error": f"Video file not found at {video_path}"}
            
        try:
            # We pass the job reference to the pipeline if it supported callbacks,
            # but for now we just block in the worker and report the final result.
            result = pipeline.run_video(path_obj, prompt=prompt, fps=fps, max_frames=max_frames)
            
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
            logger.error(f"Async Video Task Failed: {e}")
            return {"error": str(e)}
