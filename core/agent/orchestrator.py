"""
Sentinel AI — ReAct Orchestrator
Phase 10: Multi-Agent Orchestration

Implements the standard Reasoning + Acting (ReAct) Loop.
It parses LLM generated XML to intercept tool calls, executes the Python tool,
appends the observation to the context, and prompts the LLM to continue.
"""
import re
import logging
from typing import Optional

from core.agent.tools import TOOL_REGISTRY, get_tool_descriptions

logger = logging.getLogger(__name__)

class ReActOrchestrator:
    """Wraps an LLM client with an autonomous tool-calling loop."""
    
    def __init__(self, llm_client, max_iterations: int = 5):
        self.llm = llm_client
        self.max_iterations = max_iterations
        
    def _build_system_prompt(self) -> str:
        base = "You are Sentinel Agent, a multimodal reasoning system with access to external tools."
        return f"{base}\n\n{get_tool_descriptions()}\n\nUse your tools to gather facts before providing your final answer. If you do not need tools, simply provide the answer directly without XML."

    def extract_tool_call(self, text: str) -> Optional[tuple[str, str]]:
        """Parses <tool>name</tool>\n<args>...</args> from the LLM output."""
        tool_match = re.search(r"<tool>(.*?)</tool>", text, re.IGNORECASE | re.DOTALL)
        args_match = re.search(r"<args>(.*?)</args>", text, re.IGNORECASE | re.DOTALL)
        
        if tool_match and args_match:
            return tool_match.group(1).strip(), args_match.group(1).strip()
        return None

    def run(self, vision_description: str, user_prompt: str) -> str:
        """Executes the Agent autonomous loop."""
        
        system_prompt = self._build_system_prompt()
        
        # Initial prompt layout
        conversation_history = (
            f"Context from Vision Model: {vision_description}\n"
            f"User Question: {user_prompt}\n\n"
            "Begin reasoning.\n"
        )
        
        logger.info("🤖 Starting ReAct Agent Loop...")
        for iteration in range(self.max_iterations):
            logger.info(f"🤖 ReAct Iteration {iteration + 1}/{self.max_iterations}")
            
            # Request the LLM to generate the next segment
            llm_response = self.llm.generate(prompt=conversation_history, system_prompt=system_prompt)
            conversation_history += llm_response
            
            # Check if the LLM attempted to call a tool
            tool_call = self.extract_tool_call(llm_response)
            
            if tool_call:
                tool_name, tool_args = tool_call
                
                if tool_name in TOOL_REGISTRY:
                    logger.info(f"🤖 LLM Requested Tool: {tool_name}({tool_args})")
                    # Execute python function
                    try:
                        observation = TOOL_REGISTRY[tool_name](tool_args)
                    except Exception as e:
                        observation = f"Tool execution failed: {e}"
                        
                    # Inject Observation
                    observation_block = f"\n<observation>\n{observation}\n</observation>\n"
                    conversation_history += observation_block
                    logger.info("🤖 Observation injected. Prompting LLM to continue...")
                    
                    # Loop continues, allowing LLM to read the observation and generate again
                else:
                    err = f"\n<observation>\nError: Tool '{tool_name}' not found.</observation>\n"
                    conversation_history += err
            else:
                # No tool was called; the LLM believes it is finished answering.
                logger.info("🤖 ReAct Loop Complete: Final Answer Reached.")
                # We strip any internal reasoning XML if it leaked, but for now just returning the raw chat history
                # Or we can just extract the part after the last observation. We will return the whole chain for transparency.
                return conversation_history
                
        # Hit max iterations
        logger.warning("🤖 ReAct Loop interrupted: Hit max iterations. Forcing exit.")
        return conversation_history + "\n\n(Agent stopped due to reaching iteration limits.)"
