import logging
import requests
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any
from core.memory.episodic import EpisodicMemory

logger = logging.getLogger(__name__)

memory_router = APIRouter(prefix="/memory", tags=["Hive Mind"])

class SyncRequest(BaseModel):
    peer_urls: List[str]

class IngestPayload(BaseModel):
    episodes: List[Dict[str, Any]]

@memory_router.post("/ingest")
async def ingest_memory(payload: IngestPayload):
    """
    Absorbs remote embeddings sent from foreign Edge Nodes directly into the local vector space.
    This creates an omnipresent 'Hive Mind' allowing Sentinel to identify intruders seen earlier on other continents.
    """
    mem = EpisodicMemory()
    mem.import_memory(payload.episodes)
    return {"status": "success", "absorbed": len(payload.episodes)}

@memory_router.post("/sync")
async def sync_to_peers(req: SyncRequest):
    """
    Pushes the active local ChromaDB episodic graphs natively to an explicit list of peer Sentinel instances.
    """
    mem = EpisodicMemory()
    episodes = mem.export_memory()
    success_count = 0
    
    for url in req.peer_urls:
        try:
            # Strip trailing slash just in case
            target = url.rstrip('/')
            logger.info(f"HiveMind: Transmitting memory graph mapping to cluster [{target}]...")
            res = requests.post(f"{target}/memory/ingest", json={"episodes": episodes}, timeout=10)
            if res.status_code == 200:
                success_count += 1
        except Exception as e:
            logger.error(f"HiveMind: Synchronization to peer {url} failed severely: {e}")
            
    return {"status": "success", "peers_synced": success_count, "payload_size": len(episodes)}
