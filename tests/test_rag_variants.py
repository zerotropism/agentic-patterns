"""The three RAG variants: the same search, two pipelines and one agent."""

import json

import pytest
from fastmcp import Client
from ollama import ChatResponse, Message

from agentic_patterns.models import GenerationSettings
from agentic_patterns.patterns.rag import VARIANTS, raw, with_mcp
from agentic_patterns.patterns.rag.raw import RawRag, build_messages
from agentic_patterns.patterns.rag.task import BM25, Document, Question, RagTask, format_documents
from agentic_patterns.patterns.rag.with_langchain import LangChainRag
from agentic_patterns.patterns.rag.with_mcp import McpRag, build_server
from agentic_patterns.protocols import Pattern

DOCS = [
    Document("p0", "Trail Shoes", "Approx. weight: 1 lb. 1 oz. per pair."),
    Document("p1", "Wool Socks", "Weight: 3 oz. Made in USA."),
]
TASK = RagTask(
    instructions="Answer from the descriptions.",
    agent_instructions="Use the search tool.",
    k=2,
    index=BM25(DOCS),
)
QUESTION = Question("weight-p1", "weight", "How much does the Wool Socks weigh?", "p1", "3 oz")
SETTINGS = GenerationSettings(temperature=0.2, think=False)


class FakeClient:
    """Asks for one search when tools are offered and none was made yet, then answers."""

    def __init__(self, query: str = "wool socks") -> None:
        self.query = query
        self.sent: list[dict] = []

    async def chat(self, **kwargs):
        # A copy: the variant goes on appending to the list it sent
        self.sent.append(kwargs | {"messages": list(kwargs["messages"])})
        messages = kwargs["messages"]
        searched = any(
            (m.get("role") if isinstance(m, dict) else m.role) == "tool" for m in messages
        )
        if kwargs.get("tools") and not searched:
            call = Message.ToolCall(
                function=Message.ToolCall.Function(name="search", arguments={"query": self.query})
            )
            message = Message(role="assistant", content="", tool_calls=[call])
        else:
            message = Message(role="assistant", content="They weigh 3 oz.")
        reply = dict(message=message, prompt_eval_count=100, eval_count=5)
        if kwargs.get("stream"):

            async def stream():
                yield ChatResponse(model="m", done=True, done_reason="stop", **reply)

            return stream()
        return ChatResponse(**reply)


def use(monkeypatch: pytest.MonkeyPatch, client: FakeClient) -> FakeClient:
    monkeypatch.setattr(raw.ollama, "AsyncClient", lambda: client)
    monkeypatch.setattr(with_mcp.ollama, "AsyncClient", lambda: client)
    return client


async def test_raw_searches_with_the_question_then_asks_once(monkeypatch) -> None:
    client = use(monkeypatch, FakeClient())
    result = await RawRag("m", TASK, settings=SETTINGS).run(QUESTION)

    (request,) = client.sent
    context = format_documents(TASK.index.search(QUESTION.text, 2))
    assert request["messages"] == build_messages(TASK.instructions, context, QUESTION.text)
    (search,) = result.tool_calls
    assert search.arguments == {"query": QUESTION.text, "k": 2}
    assert json.loads(search.result)[0] == "p1"
    assert (result.model_calls, result.prompt_tokens) == (1, 100)


async def test_langchain_sends_what_raw_sends(monkeypatch) -> None:
    raw_client = use(monkeypatch, FakeClient())
    raw_result = await RawRag("m", TASK, settings=SETTINGS).run(QUESTION)

    client = FakeClient()
    rag = LangChainRag("m", TASK, settings=SETTINGS)
    rag.chat_model._async_client = client
    result = await rag.run(QUESTION)

    (request,) = client.sent
    sent = [{"role": m["role"], "content": m["content"]} for m in request["messages"]]
    assert sent == raw_client.sent[0]["messages"]
    assert (request["options"]["temperature"], request["think"]) == (0.2, False)
    assert result.tool_calls == raw_result.tool_calls
    assert (result.answer, result.prompt_tokens) == ("They weigh 3 oz.", 100)


async def test_mcp_lets_the_model_write_the_query(monkeypatch) -> None:
    client = use(monkeypatch, FakeClient(query="wool socks weight"))
    result = await McpRag("m", TASK, settings=SETTINGS).run(QUESTION)

    first, second = client.sent
    assert first["messages"][0] == {"role": "system", "content": TASK.agent_instructions}
    assert [t["function"]["name"] for t in first["tools"]] == ["search"]
    # The model reads the same blocks raw would have put in its prompt
    assert second["messages"][-1]["content"] == format_documents(
        TASK.index.search("wool socks weight", 2)
    )
    (search,) = result.tool_calls
    assert search.arguments == {"query": "wool socks weight", "k": 2}
    assert (result.answer, result.model_calls, result.prompt_tokens) == ("They weigh 3 oz.", 2, 200)


async def test_the_search_tool_takes_a_query_only() -> None:
    async with Client(build_server(TASK.index, 2)) as client:
        (tool,) = await client.list_tools()
        found = (await client.call_tool("search", {"query": "trail shoes"})).data

    assert tool.input_schema["required"] == ["query"]
    assert list(tool.input_schema["properties"]) == ["query"]
    assert found["ids"][0] == "p0"


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_every_variant_is_a_pattern(variant: str) -> None:
    assert isinstance(VARIANTS[variant]("m", TASK), Pattern)
