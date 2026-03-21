import logging
from fastapi import APIRouter
from core.fleet import fleet_manager

logger = logging.getLogger(__name__)

fleet_router = APIRouter(prefix="/fleet", tags=["Fleet Telemetry"])

@fleet_router.get("/status")
async def get_fleet_status():
    """Returns the total global array of physically attached Edge Node hardware limits."""
    return {"nodes": fleet_manager.get_fleet_status(), "total_active": len([n for n in fleet_manager.get_fleet_status() if n['status'] == 'ONLINE'])}
