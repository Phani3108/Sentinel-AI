import base64
import logging
from typing import Any, Dict

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.config import get_settings
from core.swarm.state import SwarmState

logger = logging.getLogger(__name__)

def get_swarm_llm():
    settings = get_settings()
    return ChatOllama(model=settings.ollama_llm_model, temperature=0.1)

def director_node(state: SwarmState) -> Dict[str, Any]:
    """
    The Orchestrator. Evaluates the current state matrices to determine if 
    insufficient data requires consulting the sub-agents before finalizing a verdict.
    """
    from core.memory.episodic import EpisodicMemory
    llm = get_swarm_llm()
    tripwire = state.get("tripwire", "Analyze threat environment.")
    vision = state.get("vision_context", "")
    intel = state.get("intel_context", "")
    messages = list(state.get("messages", []))
    
    # Retrieve Historical Memory matches to inject physical temporal awareness
    memory_node = EpisodicMemory()
    history = memory_node.recall_history(tripwire) if tripwire else ""
    
    system_prompt = f"""You are the SWARM DIRECTOR.
Your objective is to determine if the visual feed breaches the security tripwire: '{tripwire}'.

Available Intelligence:
- Vision Context: {vision if vision else 'NONE (Needs Vision Agent)'}
- Cyber Intel: {intel if intel else 'NONE (Needs Intel Agent)'}

Historical Episodic Memory (Similar past incidents):
{history if history else 'No historical matches. Context is isolated.'}

RULES:
1. If Vision Context is NONE, respond exactly with "ROUTE_TO_VISION".
2. If Vision Context exists but Cyber Intel is NONE, respond exactly with "ROUTE_TO_INTEL".
3. If both exist, formulate your final Threat Level ('CRITICAL', 'WARNING', 'CLEAR') based on the Intel constraints mapped against the Vision findings. Respond:
THREAT: [level]
REASON: [your deep technical reasoning]
"""
    
    logger.info("Swarm: Director evaluating payload.")
    response = llm.invoke([SystemMessage(content=system_prompt)] + messages)
    ans = response.content.strip()
    
    if "ROUTE_TO_VISION" in ans:
        return {"decision": "needs_vision"}
    if "ROUTE_TO_INTEL" in ans:
        return {"decision": "needs_intel"}
        
    threat = "CLEAR"
    if "THREAT: CRITICAL" in ans: threat = "CRITICAL"
    elif "THREAT: WARNING" in ans: threat = "WARNING"
    
    return {
        "decision": "finalized",
        "threat_level": threat,
        "messages": [AIMessage(content=f"[DIRECTOR]: Incident finalized as {threat}. Reason: {ans}")]
    }

def vision_node(state: SwarmState) -> Dict[str, Any]:
    """
    The Specialist. Interfaces directly with the Multimodal projection manifolds to 
    extract semantic topology from Raw Vision Bytes.
    """
    settings = get_settings()
    image_path = state.get("image_path")
    
    if not image_path:
        return {
            "vision_context": "ERROR: No image supplied.", 
            "messages": [AIMessage(content="[VISION_AGENT]: Execution failed. No image payload provided.")]
        }
        
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
        
    vision_llm = ChatOllama(model=settings.default_vision_model, temperature=0.1)
    
    msg = HumanMessage(
        content=[
            {"type": "text", "text": "Detail every object, person, anomaly, and action occurring in this image with extreme deterministic accuracy."},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_b64}"}
        ]
    )
    
    logger.info("Swarm: Vision Specialist interrogating frame manifold.")
    try:
        response = vision_llm.invoke([msg])
        output = response.content.strip()
    except Exception as e:
        output = f"Vision Model pipeline failure: {e}"

    return {
        "vision_context": output,
        "messages": [AIMessage(content=f"[VISION_AGENT]: Extracted frame topology: {output}")]
    }

def intel_node(state: SwarmState) -> Dict[str, Any]:
    """
    The Intelligence Officer. Projects vision findings against the ChromaDB Policy Graph 
    to retrieve strict alignment constraints.
    """
    vision_ctx = state.get("vision_context", "")
    from core.rag import SentinelRetriever
    
    logger.info("Swarm: Cyber Intel Agent scraping ChromaDB policy matrix.")
    try:
        retriever = SentinelRetriever()
        docs = retriever.retrieve(vision_ctx, top_k=2)
        intel = "\n".join([d["page_content"] for d in docs])
        if not intel:
            intel = "No internal security policies triggered by this visual context."
    except Exception as e:
        intel = f"Database cluster offline or empty: {e}"
        
    return {
        "intel_context": intel,
        "messages": [AIMessage(content=f"[INTEL_AGENT]: Semantic retrieval completed. Policies found: {intel}")]
    }

def action_node(state: SwarmState) -> Dict[str, Any]:
    """
    The Action Engine. Automatically triggers internal/external security mitigations 
    such as Slack Webhooks or API shutdowns based on the Swarm's consensus.
    """
    from core.integrations.webhooks import ActionEngine
    from core.memory.episodic import EpisodicMemory
    from core.security.ledger import ImmutableLedger
    from core.security.containment import containment_manager
    
    threat_level = state.get("threat_level", "UNKNOWN")
    vision_context = state.get("vision_context", "Undefined")
    tripwire = state.get("tripwire", "Undefined")
    
    # Simulate a standard "Operator" UI user lacking extreme clearance
    user_role = "OPERATOR" 
    
    logger.info("Swarm: Action Agent executing autonomous remediation strategies.")
    
    # Persist the event to Long-term Memory for future Swarm queries
    mem = EpisodicMemory()
    mem.store_episode(entity_description=vision_context, threat_level=threat_level)
    
    # Cryptographically seal this Swarm decision to prevent tampering
    ledger = ImmutableLedger()
    ledger.record_event("SWARM_DECISION", {"threat_level": threat_level, "tripwire": tripwire, "vision_ctx": vision_context})
    
    if threat_level == "CRITICAL":
        payload = f"*Tripwire Breached:* {tripwire}\n*Topology:* {vision_context}"
        
        if user_role == "ADMIN":
            res = ActionEngine.dispatch_slack_alert(message=payload, severity="CRITICAL")
            msg = f"[ACTION_AGENT]: Dispatched CRITICAL payload to Enterprise Webhook Hub."
        else:
            # Enforce Human-In-The-Loop strictness
            incident_id = containment_manager.lock_incident(payload)
            msg = f"[CONTAINMENT_LOCK]: Operator lacks authorization. Swarm halted. Incident [{incident_id}] locked. Awaiting explicit Admin Override."
            return {"messages": [AIMessage(content=msg)], "incident_id": incident_id}
            
    else:
        msg = f"[ACTION_AGENT]: Escalation protocol aborted. Threat level '{threat_level}' does not meet CRITICAL threshold."
        
    return {
        "messages": [AIMessage(content=msg)]
    }
