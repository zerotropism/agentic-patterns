"""The three judges: same rubric, same verdict type, three ways of getting it."""

import json

import pytest
from fastmcp import Client
from ollama import ChatResponse, Message

from agentic_patterns.config import load_config
from agentic_patterns.models import GenerationSettings
from agentic_patterns.patterns.judge import VARIANTS, raw, with_mcp
from agentic_patterns.patterns.judge.raw import RawJudge, build_messages
from agentic_patterns.patterns.judge.task import A, B, Verdict, judgements, load_task
from agentic_patterns.patterns.judge.with_langchain import LangChainJudge
from agentic_patterns.patterns.judge.with_mcp import PROMPT, TOOL, McpJudge, build_server
from agentic_patterns.protocols import Pattern

PROMPTS, _ = load_task(load_config())
PAIR = {
    "id": "natural-000",
    "set": "natural",
    "subset": "natural",
    "input": "Write {x} in braces.",
    "output_1": "{x}",
    "output_2": "x",
    "label": 1,
}
ITEM = judgements(PAIR)[0]
SETTINGS = GenerationSettings(temperature=0.2, think=False)


class FakeClient:
    """Answers with content, or with a tool call when tools are offered."""

    def __init__(self, content: str = '{"better": "Output (a)"}', better: str = A):
        self.content = content
        self.better = better
        self.sent: list[dict] = []

    async def chat(self, **kwargs):
        self.sent.append(kwargs)
        if kwargs.get("tools"):
            call = Message.ToolCall(
                function=Message.ToolCall.Function(name=TOOL, arguments={"better": self.better})
            )
            message = Message(role="assistant", content="", tool_calls=[call])
        else:
            message = Message(role="assistant", content=self.content)
        if kwargs.get("stream"):

            async def stream():
                yield ChatResponse(model="m", message=message, done=True, done_reason="stop")

            return stream()
        return ChatResponse(message=message)


def use(monkeypatch: pytest.MonkeyPatch, client: FakeClient) -> FakeClient:
    monkeypatch.setattr(raw.ollama, "AsyncClient", lambda: client)
    monkeypatch.setattr(with_mcp.ollama, "AsyncClient", lambda: client)
    return client


async def test_raw_constrains_the_answer_to_the_verdict(monkeypatch) -> None:
    client = use(monkeypatch, FakeClient())
    result = await RawJudge("m", PROMPTS, settings=SETTINGS).run(ITEM)

    assert result.answer == A
    (request,) = client.sent
    assert request["format"] == Verdict.model_json_schema()
    assert request["messages"] == build_messages(PROMPTS["json"], ITEM)
    assert (request["options"], request["think"]) == ({"temperature": 0.2}, False)


def test_braces_in_the_outputs_are_kept_as_written() -> None:
    (message,) = build_messages(PROMPTS["json"], ITEM)
    assert "Write {x} in braces." in message["content"] and "\n{x}\n" in message["content"]


async def test_raw_keeps_an_invalid_answer_as_written(monkeypatch) -> None:
    use(monkeypatch, FakeClient(content='{"better": "both"}'))
    assert (await RawJudge("m", PROMPTS).run(ITEM)).answer == '{"better": "both"}'


async def test_langchain_sends_the_same_request_as_raw(monkeypatch) -> None:
    raw_client = use(monkeypatch, FakeClient())
    await RawJudge("m", PROMPTS, settings=SETTINGS).run(ITEM)

    client = FakeClient(content='{"better": "Output (b)"}')
    judge = LangChainJudge("m", PROMPTS, settings=SETTINGS)
    judge.chat_model._async_client = client
    result = await judge.run(ITEM)

    (request,) = client.sent
    (raw_request,) = raw_client.sent
    sent = [{"role": m["role"], "content": m["content"]} for m in request["messages"]]
    assert sent == raw_request["messages"]
    assert request["format"] == raw_request["format"]
    assert (request["options"]["temperature"], request["think"]) == (0.2, False)
    assert result.answer == B


async def test_the_mcp_server_holds_the_rubric_and_the_verdict() -> None:
    async with Client(build_server(PROMPTS["tool"])) as client:
        (tool,) = await client.list_tools()
        prompt = await client.get_prompt(
            PROMPT, {"instruction": ITEM.instruction, "output_a": "{x}", "output_b": "x"}
        )
    assert tool.name == TOOL
    assert tool.input_schema["properties"]["better"]["enum"] == [A, B]
    assert [m.content.text for m in prompt.messages] == [
        m["content"] for m in build_messages(PROMPTS["tool"], ITEM)
    ]


async def test_mcp_returns_the_validated_verdict(monkeypatch) -> None:
    client = use(monkeypatch, FakeClient(better=B))
    result = await McpJudge("m", PROMPTS, settings=SETTINGS).run(ITEM)

    assert result.answer == B
    (request,) = client.sent
    assert request["messages"] == build_messages(PROMPTS["tool"], ITEM)
    assert [t["function"]["name"] for t in request["tools"]] == [TOOL]


async def test_mcp_refuses_a_verdict_outside_the_two_labels(monkeypatch) -> None:
    use(monkeypatch, FakeClient(better="both"))
    result = await McpJudge("m", PROMPTS).run(ITEM)
    assert result.answer.startswith("Error:")


async def test_mcp_without_a_tool_call_keeps_the_text(monkeypatch) -> None:
    class Talker(FakeClient):
        async def chat(self, **kwargs):
            return ChatResponse(message=Message(role="assistant", content="Output (a)"))

    use(monkeypatch, Talker())
    # The text happens to be a label: it counts, as raw's would
    assert (await McpJudge("m", PROMPTS).run(ITEM)).answer == A


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_every_variant_is_a_pattern(variant: str) -> None:
    assert isinstance(VARIANTS[variant]("m", PROMPTS), Pattern)


def test_the_verdict_schema_has_two_values() -> None:
    schema = Verdict.model_json_schema()
    assert schema["properties"]["better"]["enum"] == [A, B]
    assert json.loads(Verdict(better=A).model_dump_json()) == {"better": A}
