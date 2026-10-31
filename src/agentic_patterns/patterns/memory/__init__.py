"""The conversation memory use case, implemented three ways."""

from agentic_patterns.patterns.memory.raw import RawMemory
from agentic_patterns.patterns.memory.with_langchain import LangChainMemory
from agentic_patterns.patterns.memory.with_mcp import McpMemory

VARIANTS = {
    RawMemory.name: RawMemory,
    LangChainMemory.name: LangChainMemory,
    McpMemory.name: McpMemory,
}
