"""Same agent, tools served over MCP. The schemas come from the server, the loop is ours."""

import time
from collections.abc import Callable, Sequence

import ollama
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

from agentic_patterns.models import RunResult, ToolCall
from agentic_patterns.tools import TOOLS

MAX_ITERATIONS = 6


def build_server(tools: Sequence[Callable] = TOOLS, name: str = "agentic-patterns") -> FastMCP:
    """Expose plain functions as MCP tools. FastMCP derives the schema from the signature."""
    server = FastMCP(name)
    for function in tools:
        server.tool(function)
    return server


def to_ollama_tool(tool) -> dict:
    """An MCP tool already carries its JSON schema, so no reflection is needed here."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.input_schema,
        },
    }


class McpAgent:
    """The raw loop, with discovery and execution delegated to an MCP server.

    The server runs in-process here so the comparison is about the protocol, not the
    transport; a deployed server would sit behind stdio or HTTP and nothing else changes.
    """

    name = "mcp"

    def __init__(
        self,
        model: str,
        system_prompt: str,
        tools: Sequence[Callable] = TOOLS,
        max_iterations: int = MAX_ITERATIONS,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.server = build_server(tools)
        self.max_iterations = max_iterations

    async def _execute(self, client: Client, name: str, arguments: dict) -> str:
        """A failing tool reports back to the model instead of ending the run."""
        try:
            result = await client.call_tool(name, arguments)
        except ToolError as exc:
            return f"Error: {exc}"
        return result.content[0].text if result.content else ""

    async def run(self, question: str) -> RunResult:
        chat = ollama.AsyncClient()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        calls: list[ToolCall] = []
        model_calls = 0
        started = time.perf_counter()

        async with Client(self.server) as client:
            schemas = [to_ollama_tool(tool) for tool in await client.list_tools()]

            for _ in range(self.max_iterations):
                response = await chat.chat(model=self.model, messages=messages, tools=schemas)
                model_calls += 1
                message = response["message"]
                messages.append(message)

                requested = message.get("tool_calls") or []
                if not requested:
                    return RunResult(
                        variant=self.name,
                        question=question,
                        answer=message.get("content", ""),
                        tool_calls=calls,
                        model_calls=model_calls,
                        duration_seconds=time.perf_counter() - started,
                    )

                for call in requested:
                    name = call.function.name
                    arguments = dict(call.function.arguments or {})
                    result = await self._execute(client, name, arguments)
                    calls.append(ToolCall(name=name, arguments=arguments, result=result))
                    messages.append({"role": "tool", "name": name, "content": result})

        return RunResult(
            variant=self.name,
            question=question,
            answer=f"Stopped after {self.max_iterations} iterations without a final answer.",
            tool_calls=calls,
            model_calls=model_calls,
            duration_seconds=time.perf_counter() - started,
        )
