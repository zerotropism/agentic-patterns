"""Every variant sends the same model settings. No Ollama: the client is a fake."""

import pytest
from ollama import ChatResponse, Message

from agentic_patterns.models import GenerationSettings
from agentic_patterns.patterns.agent import raw, with_mcp
from agentic_patterns.patterns.agent.raw import RawAgent
from agentic_patterns.patterns.agent.with_langchain import LangChainAgent
from agentic_patterns.patterns.agent.with_mcp import McpAgent

SETTINGS = GenerationSettings(temperature=0.2, think=False)


class FakeClient:
    """Replays assistant messages and records what each chat call was sent."""

    def __init__(self, *messages: Message) -> None:
        self.replies = list(messages)
        self.sent: list[dict] = []

    async def chat(self, **kwargs) -> ChatResponse:
        self.sent.append(kwargs)
        return ChatResponse(message=self.replies.pop(0))


def tool_then_answer() -> FakeClient:
    return FakeClient(
        Message(
            role="assistant",
            tool_calls=[
                Message.ToolCall(
                    function=Message.ToolCall.Function(
                        name="calculate", arguments={"expression": "2 * 3"}
                    )
                )
            ],
        ),
        Message(role="assistant", content="It is 6."),
    )


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    client = tool_then_answer()
    monkeypatch.setattr(raw.ollama, "AsyncClient", lambda: client)
    monkeypatch.setattr(with_mcp.ollama, "AsyncClient", lambda: client)
    return client


@pytest.mark.parametrize("variant", [RawAgent, McpAgent])
async def test_the_loop_runs_the_tool_and_returns_the_answer(variant, fake) -> None:
    result = await variant(model="m", system_prompt="p", settings=SETTINGS).run("2*3?")

    assert [(call.name, call.result) for call in result.tool_calls] == [("calculate", "6")]
    assert (result.answer, result.model_calls) == ("It is 6.", 2)


@pytest.mark.parametrize("variant", [RawAgent, McpAgent])
async def test_every_model_call_sends_the_settings(variant, fake) -> None:
    await variant(model="m", system_prompt="p", settings=SETTINGS).run("2*3?")

    assert [(sent["options"], sent["think"]) for sent in fake.sent] == [
        ({"temperature": 0.2}, False)
    ] * 2


async def test_raw_and_mcp_send_the_model_the_same_request(monkeypatch) -> None:
    """Same messages, same schemas, same settings: only the machinery differs."""
    requests = []
    for variant in (RawAgent, McpAgent):
        client = tool_then_answer()
        monkeypatch.setattr(raw.ollama, "AsyncClient", lambda client=client: client)
        monkeypatch.setattr(with_mcp.ollama, "AsyncClient", lambda client=client: client)
        await variant(model="m", system_prompt="p", settings=SETTINGS).run("2*3?")
        requests.append(client.sent[0])

    assert requests[0] == requests[1]


def test_langchain_passes_temperature_and_think_to_chat_ollama() -> None:
    agent = LangChainAgent(model="m", system_prompt="p", settings=SETTINGS)

    # reasoning is how ChatOllama spells Ollama's think flag
    assert (agent.chat_model.temperature, agent.chat_model.reasoning) == (0.2, False)
