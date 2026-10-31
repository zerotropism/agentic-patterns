"""Same parsing through MCP: the type is the input schema of a tool. The model calls
record_listing with the figures, and FastMCP validates the arguments before the tool runs.
"""

import json
import time
from typing import Annotated

import ollama
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

from agentic_patterns.models import DEFAULT_SETTINGS, GenerationSettings, RunResult
from agentic_patterns.patterns.agent.with_mcp import to_ollama_tool
from agentic_patterns.patterns.parsing.raw import build_messages
from agentic_patterns.patterns.parsing.task import Item, Listing

TOOL = "record_listing"
F = Listing.model_fields


def build_server(name: str = "agentic-patterns-parsing") -> FastMCP:
    """One tool whose parameters are the fields of Listing, with the same types, bounds and
    descriptions: the tool's input schema is the type the model must fill.
    """
    server = FastMCP(name)

    @server.tool(name=TOOL)
    def record_listing(
        actual_price: Annotated[float, F["actual_price"]],
        discounted_price: Annotated[float, F["discounted_price"]],
        discount_percent: Annotated[int, F["discount_percent"]],
        rating: Annotated[float, F["rating"]],
        rating_count: Annotated[int, F["rating_count"]],
    ) -> Listing:
        """Record the figures read from one product listing."""
        return Listing(
            actual_price=actual_price,
            discounted_price=discounted_price,
            discount_percent=discount_percent,
            rating=rating,
            rating_count=rating_count,
        )

    return server


class McpParser:
    name = "mcp"

    def __init__(
        self,
        model: str,
        instructions: str,
        settings: GenerationSettings = DEFAULT_SETTINGS,
        schema_in_prompt: bool = True,
    ) -> None:
        """schema_in_prompt changes nothing here: the tool's schema is part of the prompt."""
        self.model = model
        self.instructions = instructions
        self.settings = settings
        self.server = build_server()

    async def run(self, item: Item) -> RunResult:
        started = time.perf_counter()
        async with Client(self.server) as client:
            schemas = [to_ollama_tool(tool) for tool in await client.list_tools()]
            response = await ollama.AsyncClient().chat(
                model=self.model,
                messages=build_messages(self.instructions, item),
                tools=schemas,
                options={"temperature": self.settings.temperature},
                think=self.settings.think,
            )
            message = response["message"]
            calls = message.get("tool_calls") or []
            if not calls:
                # The model answered in text instead of calling the tool: invalid
                answer = message.get("content", "")
            else:
                # Only the first call counts; no second chance, as in the other variants
                call = calls[0]
                try:
                    result = await client.call_tool(
                        call.function.name, dict(call.function.arguments or {})
                    )
                    answer = json.dumps(result.structured_content)
                except ToolError as exc:
                    answer = f"Error: {exc}"
        return RunResult(
            variant=self.name,
            question=item.id,
            answer=answer,
            model_calls=1,
            duration_seconds=time.perf_counter() - started,
        )
