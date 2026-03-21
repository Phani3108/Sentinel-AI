from typing import Dict, Any
from datetime import datetime
import uuid
import random

class FleetManager:
    """
    Tracks all completely disconnected physical IoT Edge Nodes scaling globally.
    Maintains a dictionary of active WebSocket connections, computing their live heartbeats and physical mock coordinates.
    """
    def __init__(self):
        self.active_nodes: Dict[str, Any] = {}
        
        # Hardcode a few offline nodes just to make the enterprise dashboard look populated
        self.active_nodes["mock-1"] = {
            "id": "EDGE-1FA38C", "status": "OFFLINE", "connected_at": "N/A", "last_heartbeat": "N/A",
            "location": {"lat": 51.5074, "lng": -0.1278}, "region": "EU-WEST-2"
        }
        self.active_nodes["mock-2"] = {
            "id": "EDGE-9B2D11", "status": "OFFLINE", "connected_at": "N/A", "last_heartbeat": "N/A",
            "location": {"lat": 35.6762, "lng": 139.6503}, "region": "AP-NORTHEAST-1"
        }
        
    def register_node(self) -> str:
        node_id = str(uuid.uuid4())[:8]
        lat_offset = random.uniform(-0.05, 0.05)
        lng_offset = random.uniform(-0.05, 0.05)
        
        self.active_nodes[node_id] = {
            "id": f"EDGE-{node_id.upper()}",
            "status": "ONLINE",
            "connected_at": datetime.now().isoformat(),
            "last_heartbeat": datetime.now().isoformat(),
            # Base logic defaults to US-EAST area natively
            "location": {"lat": 40.7128 + lat_offset, "lng": -74.0060 + lng_offset},
            "region": "US-EAST-1"
        }
        return node_id
        
    def ping_node(self, node_id: str):
        if node_id in self.active_nodes:
            self.active_nodes[node_id]["last_heartbeat"] = datetime.now().isoformat()
            
    def disconnect_node(self, node_id: str):
        if node_id in self.active_nodes:
            self.active_nodes[node_id]["status"] = "OFFLINE"
            
    def get_fleet_status(self) -> list:
        return list(self.active_nodes.values())

# Singleton
fleet_manager = FleetManager()
