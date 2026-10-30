"""The structured prompt use case, implemented three ways."""

from agentic_patterns.patterns.prompt.raw import RawClassifier
from agentic_patterns.patterns.prompt.with_langchain import LangChainClassifier
from agentic_patterns.patterns.prompt.with_mcp import McpClassifier

VARIANTS = {
    RawClassifier.name: RawClassifier,
    LangChainClassifier.name: LangChainClassifier,
    McpClassifier.name: McpClassifier,
}
