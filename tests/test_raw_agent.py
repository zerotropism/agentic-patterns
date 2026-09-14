"""Raw agent: schema generation and the loop itself, with a fake model."""

from agentic_patterns.patterns.agent.raw import RawAgent, tool_schema
from agentic_patterns.tools import calculate, today


def test_schema_carries_name_description_and_required_arguments() -> None:
    schema = tool_schema(calculate)["function"]

    assert schema["name"] == "calculate"
    assert "arithmetic expression" in schema["description"]
    assert schema["parameters"]["required"] == ["expression"]


def test_a_tool_without_arguments_omits_required() -> None:
    """Matching what FastMCP serves: an empty required list is absent, not empty."""
    assert "required" not in tool_schema(today)["function"]["parameters"]


def test_unknown_tool_is_reported_to_the_model() -> None:
    """The model must be able to recover, so a bad name is an answer, not an exception."""
    agent = RawAgent(model="m", system_prompt="p")
    assert "no tool named" in agent._execute("nope", {})


def test_failing_tool_is_reported_to_the_model() -> None:
    agent = RawAgent(model="m", system_prompt="p")
    assert agent._execute("calculate", {"expression": "1 +"}).startswith("Error:")


def test_successful_tool_returns_its_value() -> None:
    agent = RawAgent(model="m", system_prompt="p")
    assert agent._execute("calculate", {"expression": "2 * 3"}) == "6"
