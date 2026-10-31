"""The three memory variants send the model the same history at every turn."""

import json

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from ollama import ChatResponse, Message

from agentic_patterns.models import GenerationSettings
from agentic_patterns.patterns.memory import VARIANTS, raw, with_mcp
from agentic_patterns.patterns.memory.raw import RawMemory
from agentic_patterns.patterns.memory.task import Conversation
from agentic_patterns.patterns.memory.with_langchain import LangChainMemory
from agentic_patterns.patterns.memory.with_mcp import McpMemory, build_server
from agentic_patterns.protocols import Pattern

SYSTEM = "Answer briefly."
SETTINGS = GenerationSettings(temperature=0.2, think=False)
CONVERSATION = Conversation(
    id="city-1-0",
    slot="city",
    distance=1,
    turns=("I live in Pune.", "My sister lives in Surat.", "In which city do I live?"),
    expected="Pune",
    distractor="Surat",
)


class FakeClient:
    """Answers each turn with the number of messages it was sent, and records them."""

    def __init__(self) -> None:
        self.sent: list[list[dict]] = []
        self.settings: list[tuple] = []

    async def chat(self, **kwargs):
        messages = [{"role": m["role"], "content": m["content"]} for m in kwargs["messages"]]
        self.sent.append(messages)
        self.settings.append((kwargs["options"]["temperature"], kwargs["think"]))
        reply = dict(
            message=Message(role="assistant", content=f"reply to {len(messages)} messages"),
            prompt_eval_count=10 * len(messages),
            eval_count=4,
        )
        if kwargs.get("stream"):

            async def stream():
                yield ChatResponse(model="m", done=True, done_reason="stop", **reply)

            return stream()
        return ChatResponse(**reply)


async def run(variant: str, monkeypatch: pytest.MonkeyPatch) -> tuple[FakeClient, object]:
    client = FakeClient()
    monkeypatch.setattr(raw.ollama, "AsyncClient", lambda: client)
    monkeypatch.setattr(with_mcp.ollama, "AsyncClient", lambda: client)
    memory = VARIANTS[variant]("m", SYSTEM, settings=SETTINGS)
    if variant == "langchain":
        memory.chat_model._async_client = client
    return client, await memory.run(CONVERSATION)


async def test_every_turn_sends_the_whole_history(monkeypatch) -> None:
    client, result = await run("raw", monkeypatch)

    assert [len(messages) for messages in client.sent] == [2, 4, 6]
    assert client.sent[-1][0] == {"role": "system", "content": SYSTEM}
    assert client.sent[-1][3] == {"role": "user", "content": "My sister lives in Surat."}
    assert (result.answer, result.model_calls, result.prompt_tokens) == (
        "reply to 6 messages",
        3,
        60,
    )


@pytest.mark.parametrize("variant", ["langchain", "mcp"])
async def test_langchain_and_mcp_send_what_raw_sends(variant, monkeypatch) -> None:
    raw_client, raw_result = await run("raw", monkeypatch)
    client, result = await run(variant, monkeypatch)

    assert client.sent == raw_client.sent
    assert client.settings == raw_client.settings == [(0.2, False)] * 3
    assert (result.answer, result.prompt_tokens) == (raw_result.answer, raw_result.prompt_tokens)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
async def test_two_runs_of_a_variant_do_not_share_a_history(variant, monkeypatch) -> None:
    """The checkpointer and the MCP server outlive a run: each run must start empty."""
    client = FakeClient()
    monkeypatch.setattr(raw.ollama, "AsyncClient", lambda: client)
    monkeypatch.setattr(with_mcp.ollama, "AsyncClient", lambda: client)
    memory = VARIANTS[variant]("m", SYSTEM)
    if variant == "langchain":
        memory.chat_model._async_client = client

    await memory.run(CONVERSATION)
    await memory.run(CONVERSATION)

    assert [len(messages) for messages in client.sent] == [2, 4, 6, 2, 4, 6]


async def test_the_mcp_server_keeps_one_history_per_session() -> None:
    async with Client(build_server()) as client:
        for session, text in (("a", "first"), ("b", "other"), ("a", "second")):
            await client.call_tool(
                "append_message", {"session": session, "role": "user", "content": text}
            )
        (content,) = await client.read_resource("history://a")

    assert [m["content"] for m in json.loads(content.text)] == ["first", "second"]


async def test_the_mcp_server_refuses_an_unknown_role() -> None:
    async with Client(build_server()) as client:
        with pytest.raises(ToolError, match="role must be one of"):
            await client.call_tool(
                "append_message", {"session": "a", "role": "system", "content": "x"}
            )


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_every_variant_is_a_pattern(variant: str) -> None:
    assert isinstance(VARIANTS[variant]("m", SYSTEM), Pattern)


def test_the_variants_are_the_three_expected() -> None:
    assert (VARIANTS["raw"], VARIANTS["langchain"], VARIANTS["mcp"]) == (
        RawMemory,
        LangChainMemory,
        McpMemory,
    )
