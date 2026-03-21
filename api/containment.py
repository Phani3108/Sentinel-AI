import logging
from fastapi import APIRouter
from pydantic import BaseModel
from core.security.containment import containment_manager

logger = logging.getLogger(__name__)

containment_router = APIRouter(prefix="/containment", tags=["RBAC Containment"])

class ApprovalRequest(BaseModel):
    incident_id: str

@containment_router.post("/approve")
async def override_lock(req: ApprovalRequest):
    """
    Human-In-The-Loop physical gateway. Unlocks the Swarm Execution Graph
    dynamically dropping the payload into the Action Engine exactly as requested.
    """
    result = containment_manager.approve_incident(req.incident_id)
    return result
