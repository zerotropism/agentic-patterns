"""The same judge through MCP: the server holds the rubric, as a prompt, and the verdict, as
a tool whose one parameter can only be "Output (a)" or "Output (b)". The client fetches the
prompt, offers the tool to the model, and FastMCP validates the call before the tool runs.
"""

import time
from typing import Annotated

import ollama
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.prompts import Message

from agentic_patterns.models import DEFAULT_SETTINGS, GenerationSettings, RunResult
from agentic_patterns.patterns.agent.with_mcp import to_ollama_tool
from agentic_patterns.patterns.judge.raw import build_messages
from agentic_patterns.patterns.judge.task import Judgement, Label, Verdict
from agentic_patterns.patterns.prompt.with_mcp import to_ollama

PROMPT = "compare_outputs"
TOOL = "record_verdict"


def build_server(prompt: str, name: str = "agentic-patterns-judge") -> FastMCP:
    server = FastMCP(name)

    @server.prompt(
        name=PROMPT, description="Ask which of two outputs better follows an instruction."
    )
    def compare_outputs(instruction: str, output_a: str, output_b: str) -> list[Message]:
        judgement = Judgement("", "", "", "", "", instruction, output_a, output_b, "")
        return [Message(m["content"], role=m["role"]) for m in build_messages(prompt, judgement)]

    @server.tool(name=TOOL)
    def record_verdict(better: Annotated[Label, Verdict.model_fields["better"]]) -> Verdict:
        """Record which output better follows the instruction."""
        return Verdict(better=better)

    return server


class McpJudge:
    name = "mcp"

    def __init__(
        self, model: str, prompts: dict[str, str], settings: GenerationSettings = DEFAULT_SETTINGS
    ) -> None:
        self.model = model
        self.settings = settings
        self.server = build_server(prompts["tool"])

    async def run(self, judgement: Judgement) -> RunResult:
        started = time.perf_counter()
        async with Client(self.server) as client:
            prompt = await client.get_prompt(
                PROMPT,
                {
                    "instruction": judgement.instruction,
                    "output_a": judgement.output_a,
                    "output_b": judgement.output_b,
                },
            )
            schemas = [to_ollama_tool(tool) for tool in await client.list_tools()]
            response = await ollama.AsyncClient().chat(
                model=self.model,
                messages=to_ollama(prompt.messages),
                tools=schemas,
                options={"temperature": self.settings.temperature},
                think=self.settings.think,
            )
            message = response["message"]
            calls = message.get("tool_calls") or []
            if not calls:
                # The model answered in text instead of calling the tool: invalid
                answer = message.get("content", "")
            else:
                # Only the first call counts; no second chance, as in the other variants
                call = calls[0]
                try:
                    result = await client.call_tool(
                        call.function.name, dict(call.function.arguments or {})
                    )
                    answer = result.structured_content["better"]
                except ToolError as exc:
                    answer = f"Error: {exc}"
        return RunResult(
            variant=self.name,
            question=judgement.id,
            answer=answer,
            model_calls=1,
            duration_seconds=time.perf_counter() - started,
        )
