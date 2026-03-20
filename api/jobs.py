"""
Sentinel AI — Async Job API Routes
Phase 8: High Availability & Scaling
Provides FastAPI routes to submit and poll long-running background tasks.
"""
import os
import shutil
import tempfile
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, Form, Depends, HTTPException
from fastapi.responses import JSONResponse

from api.security.auth import get_current_user, UserAccount
from core.audit.logger import get_audit_logger
from core.worker import celery_app

router = APIRouter(prefix="/jobs", tags=["Async Jobs"])

JOBS_STORAGE_DIR = Path("data/jobs_staging")
JOBS_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/video")
async def submit_video_job(
    file: UploadFile = File(..., description="Heavy video to process asynchronously"),
    prompt: str = Form(default="Describe what is happening in this video."),
    fps: float = Form(default=0.5),
    max_frames: int = Form(default=30),
    user: UserAccount = Depends(get_current_user)
):
    """
    Submits a heavy video for background analysis. 
    Returns a Job ID to poll instead of blocking HTTP connections.
    """
    if not celery_app:
        raise HTTPException(status_code=501, detail="Celery architecture not installed/enabled.")

    from core.security.masking import get_pii_masker
    redacted_prompt, _ = get_pii_masker().redact(prompt)

    # We must save the uploaded file to a persistent staging area so the Celery worker can access it.
    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    staging_path = JOBS_STORAGE_DIR / f"{os.urandom(8).hex()}{suffix}"
    
    with open(staging_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Submit to the message broker
        task = celery_app.send_task(
            "sentinel.analyze_video",
            args=[str(staging_path), redacted_prompt, fps, max_frames]
        )
        
        get_audit_logger().log_request(
            username=user.username,
            endpoint="/jobs/video",
            prompt=redacted_prompt,
            response_meta={"job_id": task.id, "staging_file": str(staging_path)}
        )
        
        return {
            "job_id": task.id,
            "status": "PENDING",
            "message": "Video queued for background processing."
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/{job_id}")
async def get_job_status(job_id: str, user: UserAccount = Depends(get_current_user)):
    """
    Poll the status of an asynchronous job (PENDING, SUCCESS, FAILURE).
    """
    if not celery_app:
        raise HTTPException(status_code=501, detail="Celery architecture not installed/enabled.")
        
    from celery.result import AsyncResult
    res = AsyncResult(job_id, app=celery_app)
    
    response = {
        "job_id": job_id,
        "status": res.status,
    }
    
    if res.status == "SUCCESS":
        response["result"] = res.result
    elif res.status == "FAILURE":
        response["error"] = str(res.result)
        
    return response
