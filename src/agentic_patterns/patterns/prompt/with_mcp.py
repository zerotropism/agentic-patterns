"""Same prompt, served by an MCP server. The template lives on the server; the client asks
for it by name, with string arguments, and sends what it gets back to the model.
"""

import time

import ollama
from fastmcp import Client, FastMCP
from fastmcp.prompts import Message

from agentic_patterns.models import DEFAULT_SETTINGS, GenerationSettings, RunResult
from agentic_patterns.patterns.prompt.raw import build_messages
from agentic_patterns.patterns.prompt.task import Product, PromptSpec

PROMPT = "classify_product"


def build_server(spec: PromptSpec, name: str = "agentic-patterns-prompts") -> FastMCP:
    """One prompt, with two arguments. The server renders it, so a client needs nothing but
    its name: the template can change without any client changing.

    The rendering itself is raw's: what MCP changes is where it runs.
    """
    server = FastMCP(name)

    @server.prompt(name=PROMPT, description="Ask for the shop category of one product.")
    def classify_product(name: str, description: str) -> list[Message]:
        product = Product(id="", name=name, description=description, category="")
        return [Message(m["content"], role=m["role"]) for m in build_messages(spec, product)]

    return server


def to_ollama(messages) -> list[dict[str, str]]:
    """MCP prompt messages carry typed content; Ollama wants role and text."""
    return [{"role": m.role, "content": m.content.text} for m in messages]


class McpClassifier:
    name = "mcp"

    def __init__(
        self, model: str, spec: PromptSpec, settings: GenerationSettings = DEFAULT_SETTINGS
    ) -> None:
        self.model = model
        self.spec = spec
        self.settings = settings
        self.server = build_server(spec)

    async def messages(self, client: Client, product: Product) -> list[dict[str, str]]:
        result = await client.get_prompt(
            PROMPT, {"name": product.name, "description": product.description}
        )
        return to_ollama(result.messages)

    async def run(self, product: Product) -> RunResult:
        started = time.perf_counter()
        async with Client(self.server) as client:
            messages = await self.messages(client, product)
        response = await ollama.AsyncClient().chat(
            model=self.model,
            messages=messages,
            options={"temperature": self.settings.temperature},
            think=self.settings.think,
        )
        return RunResult(
            variant=self.name,
            question=product.id,
            answer=response["message"].get("content", ""),
            model_calls=1,
            duration_seconds=time.perf_counter() - started,
        )
