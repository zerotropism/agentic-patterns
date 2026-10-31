"""The three variants of typed parsing: same text, same type, three ways of enforcing it."""

import pytest
from fastmcp import Client
from ollama import ChatResponse, Message

from agentic_patterns.models import GenerationSettings
from agentic_patterns.patterns.parsing import VARIANTS, raw, with_mcp
from agentic_patterns.patterns.parsing.raw import RawParser, build_messages
from agentic_patterns.patterns.parsing.task import Item, Listing, with_schema
from agentic_patterns.patterns.parsing.with_langchain import LangChainParser
from agentic_patterns.patterns.parsing.with_mcp import TOOL, McpParser, build_server
from agentic_patterns.protocols import Pattern

TRUTH = Listing(
    actual_price=1099, discounted_price=399, discount_percent=64, rating=4.2, rating_count=24269
)
ITEM = Item("K", "was_now", "Kettle. Was ₹1,099, now ₹399 (64% off). Rated 4.2 by 24,269.", TRUTH)
SETTINGS = GenerationSettings(temperature=0.2, think=False)
INSTRUCTIONS = "Read the figures."


class FakeClient:
    """Answers with content, or with a tool call when tools are offered."""

    def __init__(self, content: str = TRUTH.model_dump_json(), arguments: dict | None = None):
        self.content = content
        self.arguments = TRUTH.model_dump() if arguments is None else arguments
        self.sent: list[dict] = []

    async def chat(self, **kwargs):
        self.sent.append(kwargs)
        if kwargs.get("tools"):
            call = Message.ToolCall(
                function=Message.ToolCall.Function(name=TOOL, arguments=self.arguments)
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


async def test_raw_validates_the_answer_against_the_type(monkeypatch) -> None:
    client = use(monkeypatch, FakeClient())
    result = await RawParser("m", INSTRUCTIONS, settings=SETTINGS).run(ITEM)

    assert Listing.model_validate_json(result.answer) == TRUTH
    (request,) = client.sent
    assert request["format"] == Listing.model_json_schema()
    assert (request["options"], request["think"]) == ({"temperature": 0.2}, False)


def test_the_schema_written_in_the_prompt_carries_the_descriptions() -> None:
    text = with_schema(INSTRUCTIONS)
    assert text.startswith(INSTRUCTIONS + "\n\n")
    assert "Price before discount" in text and '"rating_count"' in text


async def test_without_the_schema_in_the_prompt_raw_sends_the_instructions_alone(
    monkeypatch,
) -> None:
    client = use(monkeypatch, FakeClient())
    await RawParser("m", INSTRUCTIONS, schema_in_prompt=False).run(ITEM)
    assert client.sent[0]["messages"] == build_messages(INSTRUCTIONS, ITEM)


def test_langchain_can_send_the_instructions_alone() -> None:
    assert LangChainParser("m", INSTRUCTIONS, schema_in_prompt=False).instructions == INSTRUCTIONS


async def test_raw_keeps_an_invalid_answer_as_written(monkeypatch) -> None:
    use(monkeypatch, FakeClient(content='{"actual_price": "1,099"}'))
    result = await RawParser("m", INSTRUCTIONS).run(ITEM)
    assert result.answer == '{"actual_price": "1,099"}'


async def test_langchain_sends_the_same_messages_and_schema_as_raw(monkeypatch) -> None:
    raw_client = use(monkeypatch, FakeClient())
    await RawParser("m", INSTRUCTIONS, settings=SETTINGS).run(ITEM)

    client = FakeClient()
    parser = LangChainParser("m", INSTRUCTIONS, settings=SETTINGS)
    parser.chat_model._async_client = client
    result = await parser.run(ITEM)

    (request,) = client.sent
    (raw_request,) = raw_client.sent
    sent = [{"role": m["role"], "content": m["content"]} for m in request["messages"]]
    assert sent == raw_request["messages"] == build_messages(with_schema(INSTRUCTIONS), ITEM)
    assert request["format"] == raw_request["format"]
    assert (request["options"]["temperature"], request["think"]) == (0.2, False)
    assert Listing.model_validate_json(result.answer) == TRUTH


async def test_the_mcp_tool_schema_is_the_type() -> None:
    async with Client(build_server()) as client:
        (tool,) = await client.list_tools()

    expected = Listing.model_json_schema()
    properties = tool.input_schema["properties"]
    assert tool.name == TOOL
    assert set(properties) == set(expected["properties"])
    assert tool.input_schema["required"] == expected["required"]
    for name, field in expected["properties"].items():
        assert {k: v for k, v in field.items() if k != "title"} == properties[name]


async def test_mcp_returns_the_validated_arguments(monkeypatch) -> None:
    client = use(monkeypatch, FakeClient())
    result = await McpParser("m", INSTRUCTIONS, settings=SETTINGS).run(ITEM)

    assert Listing.model_validate_json(result.answer) == TRUTH
    (request,) = client.sent
    # The tool carries the schema: the instructions go alone
    assert request["messages"] == build_messages(INSTRUCTIONS, ITEM)
    assert [t["function"]["name"] for t in request["tools"]] == [TOOL]


async def test_mcp_refuses_arguments_outside_the_type(monkeypatch) -> None:
    use(monkeypatch, FakeClient(arguments=TRUTH.model_dump() | {"rating": 7}))
    result = await McpParser("m", INSTRUCTIONS).run(ITEM)

    assert result.answer.startswith("Error:") and "rating" in result.answer


async def test_mcp_without_a_tool_call_keeps_the_text(monkeypatch) -> None:
    class Talker(FakeClient):
        async def chat(self, **kwargs):
            return ChatResponse(message=Message(role="assistant", content="It costs 399."))

    use(monkeypatch, Talker())
    assert (await McpParser("m", INSTRUCTIONS).run(ITEM)).answer == "It costs 399."


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_every_variant_is_a_pattern(variant: str) -> None:
    assert isinstance(VARIANTS[variant]("m", INSTRUCTIONS), Pattern)
