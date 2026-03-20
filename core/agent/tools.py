"""
Sentinel AI — Agentic Tool Registry
Phase 10: Multi-Agent Orchestration

Provides concrete python functions that the LLM ReAct Agent can invoke to
retrieve external context grounded in reality rather than hallucinating.
"""
import sqlite3
import logging
from typing import Dict, Any, Callable

logger = logging.getLogger(__name__)

def web_search(query: str) -> str:
    """
    Mock integration for a web search engine.
    In a true enterprise scenario, this wraps DuckDuckGo, Bing, or Google Search APIs.
    """
    logger.info(f"🔧 Tool Executed: web_search('{query}')")
    
    # Very basic static mock knowledge base to simulate web context retrieval.
    mock_web_index = {
        "rosetta stone": "The Rosetta Stone is a granodiorite stele inscribed with three versions of a decree issued in Memphis, Egypt, in 196 BC.",
        "eiffel tower": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
        "python": "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability.",
        "sentinel ai": "Sentinel AI is an advanced enterprise multimodal security and analysis platform.",
    }
    
    query_lower = query.lower()
    for key, content in mock_web_index.items():
        if key in query_lower:
            return f"Search Results for '{query}': {content}"
            
    return f"Search Results for '{query}': No highly relevant public context found. Proceed with general knowledge."

def query_database(sql: str) -> str:
    """
    Allows the agent to query the internal audit SQLite database.
    """
    logger.info(f"🔧 Tool Executed: query_database('{sql}')")
    try:
        # Strictly limiting to read-only queries for safety
        if "insert" in sql.lower() or "update" in sql.lower() or "delete" in sql.lower() or "drop" in sql.lower():
            return "Security Policy Violation: Agent is restricted to SELECT queries only."
            
        with sqlite3.connect("data/audit.db") as conn:
            cursor = conn.execute(sql)
            rows = cursor.fetchall()
            return f"Database returned {len(rows)} rows: {rows[:5]} {'... (truncated)' if len(rows) > 5 else ''}"
    except Exception as e:
        return f"Database SQL Execution Error: {e}"

# The Tool Registry exposed to the ReAct Orchestrator
TOOL_REGISTRY: Dict[str, Callable[[str], str]] = {
    "web_search": web_search,
    "query_database": query_database
}

def get_tool_descriptions() -> str:
    """Returns the formatted tool schema to inject into the LLM system prompt."""
    return '''
Available Tools:
1. web_search(query): Search the public internet for definitions, history, or context about an entity.
2. query_database(sql): Execute a read-only SQLite query against the 'audit_log' table (columns: id, timestamp, username, endpoint, prompt, route, status_code).

To use a tool, you MUST output valid XML exactly like this:
<tool>tool_name</tool>
<args>argument string</args>

Do not write anything else after the tool XML block until you receive the <observation> block back.
'''
