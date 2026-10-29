"""Same agent, built by LangChain 1.x. The framework writes the loop and the schemas."""

import inspect
import time
from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ToolErrorMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama

from agentic_patterns.models import DEFAULT_SETTINGS, GenerationSettings, RunResult, ToolCall
from agentic_patterns.tools import TOOLS


def extract_calls(messages: Sequence[Any]) -> list[ToolCall]:
    """Pair each requested tool call with the ToolMessage that carries its result."""
    requested: dict[str, dict] = {}
    calls: list[ToolCall] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                requested[call["id"]] = call
        elif isinstance(message, ToolMessage):
            call = requested.get(message.tool_call_id, {})
            calls.append(
                ToolCall(
                    name=message.name or call.get("name", "?"),
                    arguments=call.get("args", {}),
                    result=str(message.content),
                )
            )
    return calls


def report_tool_error(exc: Exception, request: Any) -> str:
    """The text raw and mcp send back for a failing tool, so the model can correct its call."""
    return f"Error: {exc}"


def described(function: Callable) -> StructuredTool:
    """The tool with the description raw and mcp send: the docstring, dedented.

    Given a plain function, LangChain keeps the docstring's inner indentation. Measured on
    29 October 2026, four leading spaces on the second paragraph of calculate's description
    were enough for qwen3.5:4b-mlx to stop calling today and guess the year.
    """
    return StructuredTool.from_function(function, description=inspect.getdoc(function))


class LangChainAgent:
    """No loop to write: create_agent builds the graph, binds the tools and runs them."""

    name = "langchain"

    def __init__(
        self,
        model: str,
        system_prompt: str,
        tools: Sequence[Callable] = TOOLS,
        settings: GenerationSettings = DEFAULT_SETTINGS,
        chat_model: Any = None,
        framework_defaults: bool = False,
    ) -> None:
        """framework_defaults=True drops the two adjustments below and reproduces what
        create_agent does out of the box, as measured on 21 and 29 October 2026.
        """
        self.model = model
        self.settings = settings
        # chat_model is injected by the tests; production builds a real ChatOllama.
        # reasoning is ChatOllama's name for Ollama's think flag.
        self.chat_model = chat_model or ChatOllama(
            model=model, temperature=settings.temperature, reasoning=settings.think
        )
        # Out of the box, create_agent only reports invalid arguments to the model; any
        # other exception raised by a tool ends the run. raw and mcp report every failure.
        # And the descriptions must be the ones raw and mcp send, see described().
        middleware = [] if framework_defaults else [ToolErrorMiddleware(report_tool_error)]
        bound = list(tools) if framework_defaults else [described(t) for t in tools]
        self.agent = create_agent(
            self.chat_model,
            bound,
            system_prompt=system_prompt,
            middleware=middleware,
        )

    async def run(self, question: str) -> RunResult:
        started = time.perf_counter()
        state = await self.agent.ainvoke({"messages": [{"role": "user", "content": question}]})
        messages = state["messages"]

        return RunResult(
            variant=self.name,
            question=question,
            answer=str(messages[-1].content),
            tool_calls=extract_calls(messages),
            model_calls=sum(isinstance(m, AIMessage) for m in messages),
            duration_seconds=time.perf_counter() - started,
        )
