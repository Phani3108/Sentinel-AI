"""
Tests for Phase 10 Multi-Agent Orchestration
Validates the Tool Registry and the ReAct execution loop handling LLM XML payloads.
"""
from unittest.mock import MagicMock
from core.agent.orchestrator import ReActOrchestrator
from core.agent.tools import web_search, query_database

def test_web_search_tool_mock():
    # Test valid key
    res = web_search("rosetta stone")
    assert "memphis" in res.lower()
    
    # Test invalid key
    res = web_search("unknown object 123")
    assert "no highly relevant public context found" in res.lower()

def test_database_query_tool_mock():
    # Test read-only
    res = query_database("SELECT * FROM non_existent_table")
    assert "Database SQL Execution Error" in res  # It should fail safely on non-existent table, but not policy violation
    
    # Test mutation blocked
    res = query_database("DROP TABLE audit_log")
    assert "Security Policy Violation" in res

def test_orchestrator_tool_extraction():
    # Setup mock LLM
    mock_llm = MagicMock()
    orchestrator = ReActOrchestrator(mock_llm)
    
    # Case 1: valid XML
    text = "Thinking... I need to search.\n<tool>web_search</tool>\n<args>rosetta stone</args>\nDone."
    name, args = orchestrator.extract_tool_call(text)
    assert name == "web_search"
    assert args == "rosetta stone"
    
    # Case 2: invalid or no XML
    text = "I know the answer. It is the rosetta stone."
    res = orchestrator.extract_tool_call(text)
    assert res is None

def test_orchestrator_execution_loop():
    mock_llm = MagicMock()
    # Mock sequence: 1st response asks for tool, 2nd response finalizes
    mock_llm.generate.side_effect = [
        "I need more context.\n<tool>web_search</tool>\n<args>rosetta stone</args>",
        "Now I see. The answer is the Rosetta Stone from Memphis, Egypt."
    ]
    orchestrator = ReActOrchestrator(mock_llm)
    
    final_output = orchestrator.run("A photo of an old large curved stone with text.", "What is this?")
    
    # Loop should have injected observation and called LLM twice
    assert mock_llm.generate.call_count == 2
    assert "<observation>" in final_output
    assert "Memphis" in final_output
