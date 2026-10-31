"""The typed parsing use case, implemented three ways."""

from agentic_patterns.patterns.parsing.raw import RawParser
from agentic_patterns.patterns.parsing.with_langchain import LangChainParser
from agentic_patterns.patterns.parsing.with_mcp import McpParser

VARIANTS = {
    RawParser.name: RawParser,
    LangChainParser.name: LangChainParser,
    McpParser.name: McpParser,
}
