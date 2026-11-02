"""The RAG use case, implemented three ways."""

from agentic_patterns.patterns.rag.raw import RawRag
from agentic_patterns.patterns.rag.with_langchain import LangChainRag
from agentic_patterns.patterns.rag.with_mcp import McpRag

VARIANTS = {RawRag.name: RawRag, LangChainRag.name: LangChainRag, McpRag.name: McpRag}
