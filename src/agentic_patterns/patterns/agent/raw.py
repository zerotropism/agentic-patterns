"""Agent with no framework: the tool loop, written out."""

import inspect
import time
from collections.abc import Callable, Sequence

import ollama

from agentic_patterns.models import RunResult, ToolCall
from agentic_patterns.tools import TOOLS

MAX_ITERATIONS = 6
JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


def tool_schema(function: Callable) -> dict:
    """Describe a plain function the way the model expects.

    This is the part a framework writes for you: name, description, parameter types,
    and which parameters are required. The ollama client can also accept a callable
    directly; it is spelled out here because that is what this variant is about.

    The shape matches what FastMCP serves for the same function, so the comparison
    between variants measures the approach rather than the schema.
    """
    properties, required = {}, []

    for name, parameter in inspect.signature(function).parameters.items():
        properties[name] = {"type": JSON_TYPES.get(parameter.annotation, "string")}
        if parameter.default is inspect.Parameter.empty:
            required.append(name)

    parameters: dict = {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
    }
    if required:
        parameters["required"] = required

    return {
        "type": "function",
        "function": {
            "name": function.__name__,
            "description": inspect.getdoc(function) or "",
            "parameters": parameters,
        },
    }


class RawAgent:
    """Calls the model, executes the tools it asks for, calls it again until it answers."""

    name = "raw"

    def __init__(
        self,
        model: str,
        system_prompt: str,
        tools: Sequence[Callable] = TOOLS,
        max_iterations: int = MAX_ITERATIONS,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.tools = {tool.__name__: tool for tool in tools}
        self.schemas = [tool_schema(tool) for tool in tools]
        self.max_iterations = max_iterations

    def _execute(self, name: str, arguments: dict) -> str:
        """Run one tool. A failing tool reports back to the model instead of ending the run."""
        tool = self.tools.get(name)
        if tool is None:
            return f"Error: no tool named '{name}'. Available: {sorted(self.tools)}"
        try:
            return str(tool(**arguments))
        except Exception as exc:
            return f"Error: {exc}"

    async def run(self, question: str) -> RunResult:
        client = ollama.AsyncClient()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        calls: list[ToolCall] = []
        model_calls = 0
        started = time.perf_counter()

        for _ in range(self.max_iterations):
            response = await client.chat(model=self.model, messages=messages, tools=self.schemas)
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
                result = self._execute(name, arguments)
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
