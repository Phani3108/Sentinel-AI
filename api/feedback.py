"""
Sentinel AI — Feedback API Routes
Phase 9: Continuous Learning Flywheel
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional

from api.security.auth import get_current_user, UserAccount
from core.audit.flywheel import get_flywheel
from core.audit.logger import get_audit_logger

router = APIRouter(prefix="/feedback", tags=["RLHF Flywheel"])

class FeedbackRequest(BaseModel):
    prompt: str
    original_answer: str
    rating: int # 1 for up, -1 for down
    image_hash: Optional[str] = None
    user_correction: Optional[str] = None

@router.post("/")
async def submit_feedback(
    req: FeedbackRequest,
    user: UserAccount = Depends(get_current_user)
):
    """
    Accepts human corrections on the UI and adds them to the data flywheel.
    """
    if req.rating not in (1, -1):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Rating must be 1 (Thumbs Up) or -1 (Thumbs Down)"
        )
        
    try:
        # Save to SQLite Flywheel
        get_flywheel().log_feedback(
            username=user.username,
            prompt=req.prompt,
            original_answer=req.original_answer,
            rating=req.rating,
            image_hash=req.image_hash,
            user_correction=req.user_correction
        )
        
        # Also log to primary audit trail
        get_audit_logger().log_request(
            username=user.username,
            endpoint="/feedback",
            prompt=req.prompt,
            image_hash=req.image_hash,
            response_meta={"rating": req.rating, "has_correction": bool(req.user_correction)}
        )
        
        return {"status": "success", "message": "Feedback integrated into Flywheel."}
    except Exception as e:
        return {"status": "error", "message": str(e)}
