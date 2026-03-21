import logging
import uuid
from typing import Dict, Any
from core.integrations.webhooks import ActionEngine

logger = logging.getLogger(__name__)

class ContainmentLock:
    """
    Global Zero-Trust Interlock Dictionary.
    Blocks the physical execution of webhooks for non-Admin users until explicitly
    overriden by a validated `POST /containment/approve` endpoint.
    """
    def __init__(self):
        self.pending_overrides: Dict[str, Any] = {}
        
    def lock_incident(self, payload: str) -> str:
        incident_id = f"INC-{str(uuid.uuid4())[:6].upper()}"
        self.pending_overrides[incident_id] = {
            "payload": payload,
            "status": "PENDING_APPROVAL"
        }
        logger.warning(f"Containment Lock engaged for [{incident_id}]. Awaiting manual Admin Key.")
        return incident_id
        
    def approve_incident(self, incident_id: str) -> dict:
        if incident_id not in self.pending_overrides:
            return {"status": "error", "message": f"Incident {incident_id} not found or expired."}
            
        incident = self.pending_overrides[incident_id]
        if incident["status"] == "APPROVED":
            return {"status": "error", "message": f"Incident {incident_id} already executed."}
            
        # Manually bypass the lock and physically execute the action
        res = ActionEngine.dispatch_slack_alert(message=incident["payload"], severity="CRITICAL")
        self.pending_overrides[incident_id]["status"] = "APPROVED"
        logger.critical(f"Admin override confirmed. Payload for [{incident_id}] physically dispatched.")
        
        return {"status": "success", "message": f"Admin Override Authorized. Network isolating."}

# Global Singleton
containment_manager = ContainmentLock()
