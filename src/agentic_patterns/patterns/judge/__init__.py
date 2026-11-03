"""The LLM-as-judge use case, implemented three ways."""

from agentic_patterns.patterns.judge.raw import RawJudge
from agentic_patterns.patterns.judge.with_langchain import LangChainJudge
from agentic_patterns.patterns.judge.with_mcp import McpJudge

VARIANTS = {
    RawJudge.name: RawJudge,
    LangChainJudge.name: LangChainJudge,
    McpJudge.name: McpJudge,
}
