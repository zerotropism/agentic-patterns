"""Same conversation, with the history kept by an MCP server.

The server holds one history per session: a tool appends a message, a resource serves the
history. Before each call, the client reads the resource and sends what it gets; after it,
the client appends the answer. Any client that knows the session id gets the same history.
"""

import json
import time
import uuid
from collections import defaultdict

import ollama
from fastmcp import Client, FastMCP

from agentic_patterns.models import DEFAULT_SETTINGS, GenerationSettings, RunResult
from agentic_patterns.patterns.memory.task import Conversation

ROLES = ("user", "assistant")


def build_server(name: str = "agentic-patterns-memory") -> FastMCP:
    server = FastMCP(name)
    sessions: dict[str, list[dict[str, str]]] = defaultdict(list)

    @server.tool
    def append_message(session: str, role: str, content: str) -> int:
        """Add one message to a session's history. Returns the history's new length."""
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, not '{role}'.")
        sessions[session].append({"role": role, "content": content})
        return len(sessions[session])

    @server.resource("history://{session}", mime_type="application/json")
    def history(session: str) -> str:
        """A session's messages, oldest first."""
        return json.dumps(sessions[session])

    return server


class McpMemory:
    name = "mcp"

    def __init__(
        self, model: str, system: str, settings: GenerationSettings = DEFAULT_SETTINGS
    ) -> None:
        self.model = model
        self.system = system
        self.settings = settings
        self.server = build_server()

    async def run(self, conversation: Conversation) -> RunResult:
        session = f"{conversation.id}-{uuid.uuid4()}"
        chat = ollama.AsyncClient()
        started = time.perf_counter()
        answer, prompt_tokens = "", None

        async with Client(self.server) as client:
            for turn in conversation.turns:
                await client.call_tool(
                    "append_message", {"session": session, "role": "user", "content": turn}
                )
                (content,) = await client.read_resource(f"history://{session}")
                history = json.loads(content.text)
                response = await chat.chat(
                    model=self.model,
                    messages=[{"role": "system", "content": self.system}, *history],
                    options={"temperature": self.settings.temperature},
                    think=self.settings.think,
                )
                answer = response["message"].get("content", "")
                prompt_tokens = response.get("prompt_eval_count")
                await client.call_tool(
                    "append_message",
                    {"session": session, "role": "assistant", "content": answer},
                )

        return RunResult(
            variant=self.name,
            question=conversation.id,
            answer=answer,
            model_calls=len(conversation.turns),
            duration_seconds=time.perf_counter() - started,
            prompt_tokens=prompt_tokens,
        )
