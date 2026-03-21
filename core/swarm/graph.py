from langgraph.graph import StateGraph, END
from core.swarm.state import SwarmState
from core.swarm.nodes import director_node, vision_node, intel_node, action_node

def route_director(state: SwarmState):
    """Conditional Edge Evaluation: Determine next state jump based on Orchestrator requirements."""
    decision = state.get("decision", "")
    threat_level = state.get("threat_level", "")
    if decision == "needs_vision":
        return "vision_agent"
    elif decision == "needs_intel":
        return "intel_agent"
    elif decision == "finalized" and threat_level == "CRITICAL":
        return "action_agent"
    return END

def compile_swarm():
    """Compiles the asynchronous Cylical State Graph routing mechanics into an executable runtime."""
    workflow = StateGraph(SwarmState)
    
    # Inject Neural Nodes
    workflow.add_node("director", director_node)
    workflow.add_node("vision_agent", vision_node)
    workflow.add_node("intel_agent", intel_node)
    workflow.add_node("action_agent", action_node)
    
    # Dictate graph traversal starting point
    workflow.set_entry_point("director")
    
    # Conditional logic branching
    workflow.add_conditional_edges(
        "director",
        route_director,
        {
            "vision_agent": "vision_agent",
            "intel_agent": "intel_agent",
            "action_agent": "action_agent",
            END: END
        }
    )
    
    # Forced state jumping returning to Director for evaluation
    workflow.add_edge("vision_agent", "director")
    workflow.add_edge("intel_agent", "director")
    
    # Terminal Execution
    workflow.add_edge("action_agent", END)
    
    # Compile execution engine
    return workflow.compile()
