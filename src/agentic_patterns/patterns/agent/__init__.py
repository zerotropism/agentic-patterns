"""The agent use case, implemented three ways."""

from agentic_patterns.patterns.agent.raw import RawAgent
from agentic_patterns.patterns.agent.with_langchain import LangChainAgent
from agentic_patterns.patterns.agent.with_mcp import McpAgent

# Adding a variant means adding a class and one line here
VARIANTS = {
    RawAgent.name: RawAgent,
    LangChainAgent.name: LangChainAgent,
    McpAgent.name: McpAgent,
}
