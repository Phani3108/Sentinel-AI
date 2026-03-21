import operator
from typing import Annotated, Sequence, TypedDict, Optional
from langchain_core.messages import BaseMessage

class SwarmState(TypedDict):
    """
    The rigid, cyclical state object passed continuously between the Swarm Agents.
    Contains the deterministic visual findings, intelligence policies, and debate messages.
    """
    # The conversational history array between the Director and the Specialist Nodes
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # Input payloads
    tripwire: str
    image_path: Optional[str]
    
    # Extracted Intel
    vision_context: str
    intel_context: str
    
    # Final Outcome (determined by the Director)
    threat_level: str
    decision: str
