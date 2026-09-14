"""Same agent, built by LangChain 1.x. The framework writes the loop and the schemas."""

import time
from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_ollama import ChatOllama

from agentic_patterns.models import RunResult, ToolCall
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


class LangChainAgent:
    """No loop to write: create_agent builds the graph, binds the tools and runs them."""

    name = "langchain"

    def __init__(
        self,
        model: str,
        system_prompt: str,
        tools: Sequence[Callable] = TOOLS,
        temperature: float = 0.0,
        chat_model: Any = None,
    ) -> None:
        self.model = model
        # chat_model is injected by the tests; production builds a real ChatOllama
        self.agent = create_agent(
            chat_model or ChatOllama(model=model, temperature=temperature),
            list(tools),
            system_prompt=system_prompt,
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
