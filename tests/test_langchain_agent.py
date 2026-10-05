"""LangChain variant, driven by a fake chat model: no Ollama, no network."""

from datetime import date

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from agentic_patterns.patterns.agent.with_langchain import LangChainAgent
from agentic_patterns.tools import calculate, today


class ToolCallingFake(GenericFakeChatModel):
    """GenericFakeChatModel refuses bind_tools; create_agent requires it."""

    def bind_tools(self, tools, **kwargs):
        return self


def make_agent(*responses: AIMessage) -> LangChainAgent:
    return LangChainAgent(
        model="fake",
        system_prompt="Use tools.",
        tools=[today, calculate],
        chat_model=ToolCallingFake(messages=iter(responses)),
    )


async def test_a_single_tool_call_is_executed_and_reported() -> None:
    agent = make_agent(
        AIMessage(content="", tool_calls=[{"name": "today", "args": {}, "id": "1"}]),
        AIMessage(content="Today is 2026-09-14."),
    )
    result = await agent.run("date?")

    assert result.variant == "langchain"
    assert [(c.name, c.result) for c in result.tool_calls] == [("today", date.today().isoformat())]
    assert result.answer == "Today is 2026-09-14."


async def test_arguments_are_paired_with_their_result() -> None:
    agent = make_agent(
        AIMessage(
            content="",
            tool_calls=[{"name": "calculate", "args": {"expression": "2 * 3"}, "id": "1"}],
        ),
        AIMessage(content="It is 6."),
    )
    call = (await agent.run("2*3?")).tool_calls[0]

    assert call.arguments == {"expression": "2 * 3"}
    assert call.result == "6"


async def test_two_calls_in_one_turn_are_both_reported() -> None:
    """The case llama3.2:3b gets wrong in the raw variant: two tools, one turn."""
    agent = make_agent(
        AIMessage(
            content="",
            tool_calls=[
                {"name": "today", "args": {}, "id": "1"},
                {"name": "calculate", "args": {"expression": "365 * 3"}, "id": "2"},
            ],
        ),
        AIMessage(content="Done."),
    )
    result = await agent.run("both?")

    assert [c.name for c in result.tool_calls] == ["today", "calculate"]


async def test_model_calls_counts_the_round_trips() -> None:
    agent = make_agent(
        AIMessage(content="", tool_calls=[{"name": "today", "args": {}, "id": "1"}]),
        AIMessage(content="Done."),
    )
    assert (await agent.run("date?")).model_calls == 2
