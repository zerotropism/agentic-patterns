"""LangChain variant, driven by a fake chat model: no Ollama, no network."""

from datetime import date

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from agentic_patterns.patterns.agent.raw import tool_schema
from agentic_patterns.patterns.agent.with_langchain import LangChainAgent
from agentic_patterns.tools import TOOLS, calculate, today


class ToolCallingFake(GenericFakeChatModel):
    """GenericFakeChatModel refuses bind_tools; create_agent requires it."""

    def bind_tools(self, tools, **kwargs):
        return self


def make_agent(*responses: AIMessage, framework_defaults: bool = False) -> LangChainAgent:
    return LangChainAgent(
        model="fake",
        system_prompt="Use tools.",
        tools=[today, calculate],
        chat_model=ToolCallingFake(messages=iter(responses)),
        framework_defaults=framework_defaults,
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


def bad_then_good_calculation() -> list[AIMessage]:
    return [
        AIMessage(
            content="",
            tool_calls=[{"name": "calculate", "args": {"expression": "2^10"}, "id": "1"}],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "calculate", "args": {"expression": "2**10"}, "id": "2"}],
        ),
        AIMessage(content="1024"),
    ]


async def test_by_default_a_failing_tool_ends_the_run() -> None:
    """What create_agent does out of the box: ToolNode only turns invalid-argument errors
    into messages and re-raises anything else. Measured on 21 October 2026: every run of
    the recovery question failed this way.
    """
    agent = make_agent(*bad_then_good_calculation(), framework_defaults=True)
    with pytest.raises(ValueError, match="Unsupported expression element"):
        await agent.run("2^10?")


async def test_a_failing_tool_is_reported_to_the_model_as_raw_and_mcp_do() -> None:
    result = await make_agent(*bad_then_good_calculation()).run("2^10?")

    first, second = result.tool_calls
    assert first.result.startswith("Error: Unsupported expression element")
    assert (second.result, result.answer, result.model_calls) == ("1024", "1024", 3)


def bound_descriptions(agent: LangChainAgent) -> dict[str, str]:
    tools = agent.agent.get_graph().nodes["tools"].data.tools_by_name
    return {name: tool.description for name, tool in tools.items()}


def test_descriptions_are_the_ones_raw_sends() -> None:
    """The cause measured on 29 October: indentation kept in calculate's description."""
    raw = {s["function"]["name"]: s["function"]["description"] for s in map(tool_schema, TOOLS)}
    assert bound_descriptions(make_agent()) == raw


def test_by_default_langchain_keeps_the_docstring_indentation() -> None:
    description = bound_descriptions(make_agent(framework_defaults=True))["calculate"]
    assert "\n    Supports" in description
