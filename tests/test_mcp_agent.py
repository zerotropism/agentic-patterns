"""MCP variant: server construction, schema conversion, tool execution. No Ollama."""

from fastmcp import Client

from agentic_patterns.patterns.agent.with_mcp import (
    McpAgent,
    build_server,
    to_ollama_tool,
)
from agentic_patterns.tools import TOOLS


async def test_every_shared_tool_is_exposed() -> None:
    async with Client(build_server(TOOLS)) as client:
        assert {tool.name for tool in await client.list_tools()} == {
            "today",
            "calculate",
        }


async def test_schema_comes_from_the_server_not_from_reflection() -> None:
    """Unlike the raw variant, nothing here inspects the Python signature."""
    async with Client(build_server(TOOLS)) as client:
        tools = {tool.name: to_ollama_tool(tool) for tool in await client.list_tools()}

    calculate = tools["calculate"]["function"]
    assert "arithmetic expression" in calculate["description"]
    assert calculate["parameters"]["required"] == ["expression"]


async def test_a_tool_call_returns_its_result() -> None:
    agent = McpAgent(model="m", system_prompt="p")
    async with Client(agent.server) as client:
        assert await agent._execute(client, "calculate", {"expression": "2 * 3"}) == "6"


async def test_a_failing_tool_is_reported_to_the_model() -> None:
    agent = McpAgent(model="m", system_prompt="p")
    async with Client(agent.server) as client:
        assert (await agent._execute(client, "calculate", {"expression": "1 +"})).startswith(
            "Error:"
        )


async def test_an_unknown_tool_is_reported_to_the_model() -> None:
    agent = McpAgent(model="m", system_prompt="p")
    async with Client(agent.server) as client:
        assert (await agent._execute(client, "nope", {})).startswith("Error:")
