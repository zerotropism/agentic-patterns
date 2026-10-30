"""The three variants of the structured prompt send the model the same request."""

import pytest
from fastmcp import Client
from ollama import ChatResponse, Message

from agentic_patterns.models import GenerationSettings
from agentic_patterns.patterns.prompt import VARIANTS, raw, with_mcp
from agentic_patterns.patterns.prompt.raw import RawClassifier, build_messages
from agentic_patterns.patterns.prompt.task import Product, PromptSpec
from agentic_patterns.patterns.prompt.with_langchain import LangChainClassifier, build_prompt
from agentic_patterns.patterns.prompt.with_mcp import PROMPT, McpClassifier, build_server
from agentic_patterns.protocols import Pattern

SPEC = PromptSpec(
    instructions="Sort into one of: {categories}. Answer with the category only.",
    user="Product: {name}\nDescription: {description}",
    categories=("Electronics", "Home&Kitchen"),
    examples=(
        Product("e", "HDMI cable", "4K, 3 feet", "Electronics"),
        Product("h", "Kettle", "1.5 l, {steel}", "Home&Kitchen"),
    ),
)
PRODUCT = Product("p", "Toaster", "Two slots, 800 W", "Home&Kitchen")
SETTINGS = GenerationSettings(temperature=0.2, think=False)
ROLES = {"human": "user", "ai": "assistant"}


def test_instructions_come_first_then_one_exchange_per_example() -> None:
    messages = build_messages(SPEC, PRODUCT)

    assert [m["role"] for m in messages] == [
        "user",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert messages[0]["content"].startswith("Sort into one of: Electronics, Home&Kitchen.")
    assert messages[-1]["content"] == "Product: Toaster\nDescription: Two slots, 800 W"


def test_a_brace_in_a_value_is_text_not_a_placeholder() -> None:
    assert build_messages(SPEC, PRODUCT)[3]["content"].endswith("1.5 l, {steel}")


def test_langchain_builds_the_same_messages() -> None:
    formatted = build_prompt(SPEC).format_messages(
        name=PRODUCT.name, description=PRODUCT.description
    )
    assert [{"role": ROLES[m.type], "content": m.content} for m in formatted] == build_messages(
        SPEC, PRODUCT
    )


async def test_the_mcp_server_serves_the_same_messages() -> None:
    classifier = McpClassifier("m", SPEC)
    async with Client(classifier.server) as client:
        assert await classifier.messages(client, PRODUCT) == build_messages(SPEC, PRODUCT)


async def test_the_mcp_prompt_declares_its_two_arguments() -> None:
    async with Client(build_server(SPEC)) as client:
        (prompt,) = await client.list_prompts()
    assert prompt.name == PROMPT
    assert {a.name for a in prompt.arguments} == {"name", "description"}
    assert all(a.required for a in prompt.arguments)


def test_a_missing_variable_is_caught_by_langchain_before_any_call() -> None:
    with pytest.raises(KeyError):
        build_prompt(SPEC).format_messages(name="only a name")


class FakeClient:
    def __init__(self, answer: str = "Home&Kitchen") -> None:
        self.answer = answer
        self.sent: list[dict] = []

    async def chat(self, **kwargs):
        self.sent.append(kwargs)
        message = Message(role="assistant", content=self.answer)
        if kwargs.get("stream"):

            async def stream():
                yield ChatResponse(model="m", message=message, done=True, done_reason="stop")

            return stream()
        return ChatResponse(message=message)


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    client = FakeClient()
    monkeypatch.setattr(raw.ollama, "AsyncClient", lambda: client)
    monkeypatch.setattr(with_mcp.ollama, "AsyncClient", lambda: client)
    return client


async def test_raw_and_mcp_send_the_same_request(fake) -> None:
    for variant in (RawClassifier, McpClassifier):
        result = await variant("m", SPEC, settings=SETTINGS).run(PRODUCT)
        assert (result.answer, result.question, result.model_calls) == ("Home&Kitchen", "p", 1)

    raw_request, mcp_request = fake.sent
    assert raw_request == mcp_request
    assert (raw_request["options"], raw_request["think"]) == ({"temperature": 0.2}, False)


async def test_langchain_sends_the_same_messages_and_settings() -> None:
    raw_messages = build_messages(SPEC, PRODUCT)
    client = FakeClient()
    classifier = LangChainClassifier("m", SPEC, settings=SETTINGS)
    classifier.chat_model._async_client = client

    result = await classifier.run(PRODUCT)

    (request,) = client.sent
    sent = [{"role": m["role"], "content": m["content"]} for m in request["messages"]]
    assert sent == raw_messages
    assert (request["options"]["temperature"], request["think"]) == (0.2, False)
    assert result.answer == "Home&Kitchen"


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_every_variant_is_a_pattern(variant: str) -> None:
    assert isinstance(VARIANTS[variant]("m", SPEC), Pattern)
