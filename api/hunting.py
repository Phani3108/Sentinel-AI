import sqlite3
import logging
from fastapi import APIRouter
from pydantic import BaseModel
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_community.chat_models import ChatOllama

from core.config import get_settings
from core.memory.episodic import EpisodicMemory

logger = logging.getLogger(__name__)

hunting_router = APIRouter(prefix="/hunting", tags=["Threat Hunting"])

class HuntRequest(BaseModel):
    query: str

@tool
def search_chroma_incidents(query: str) -> str:
    """Explicitly searches the Enterprise Episodic Memory temporal embeddings for specific semantic visual entities or historical incidents."""
    mem = EpisodicMemory()
    return mem.recall_history(query, top_k=7)

@tool
def query_flywheel_ledger(query: str) -> str:
    """Retrieves chronological Zero-Trust ledger logs proving if the Autonomous Swarm took explicit actions (Webhooks/Alerts)."""
    try:
        with sqlite3.connect("db_flywheel.sqlite3") as conn:
            cursor = conn.execute("SELECT timestamp, event_type, payload FROM audit_ledger ORDER BY id DESC LIMIT 10")
            rows = cursor.fetchall()
            return str(rows)
    except Exception as e:
        return f"SQL Database extraction fault: {e}"

@hunting_router.post("/query")
async def hunt_data_lake(req: HuntRequest):
    """
    Spins up an independent ReAct LangGraph analyst to aggressively interrogate 
    both the SQL and Vector databases simultaneously based on Natural Language parameters.
    """
    settings = get_settings()
    llm = ChatOllama(model=settings.ollama_llm_model, temperature=0.1)
    
    # Mount internal infrastructure tools
    tools = [search_chroma_incidents, query_flywheel_ledger]
    
    agent = create_react_agent(llm, tools=tools)
    
    system_prompt = (
        "You are the Sentinel Enterprise Threat Analyst. "
        "Interrogate the Vector (Chroma) and SQL databases using your tools to map macroscopic temporal incursions. "
        "Synthesize explicit timelines binding visual threats to autonomous Swarm remedies based strictly on the retrieved data."
        "If you do not find the requested incidents, unequivocally state the network is secure."
    )
    
    logger.info(f"Threat Hunt Initiated: {req.query}")
    try:
        messages = agent.invoke({"messages": [("system", system_prompt), ("human", req.query)]})
        final_response = messages["messages"][-1].content
    except Exception as e:
        logger.error(f"ReAct Analyst crashed: {e}")
        final_response = f"Analyst failure: {e}"
        
    return {"query": req.query, "analysis": final_response}
