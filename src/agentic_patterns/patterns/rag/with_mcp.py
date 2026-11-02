"""Agentic RAG through MCP: the search is a tool, and the model decides when to call it and
with what query. The loop is the agent case's: call the model, run the tools it asks for,
call it again, until it answers.
"""

import json
import time

import ollama
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

from agentic_patterns.models import DEFAULT_SETTINGS, GenerationSettings, RunResult, ToolCall
from agentic_patterns.patterns.agent.with_mcp import to_ollama_tool
from agentic_patterns.patterns.rag.task import BM25, Question, RagTask, format_documents

MAX_ITERATIONS = 4


def build_server(index: BM25, k: int, name: str = "agentic-patterns-rag") -> FastMCP:
    server = FastMCP(name)

    @server.tool
    def search(query: str) -> dict:
        """Search the shop's product descriptions. Returns the best matches, numbered, each
        with its name and description.
        """
        documents = index.search(query, k)
        # text is what the model reads, the same blocks raw puts in its prompt; ids is for
        # the client, which records what was found
        return {"ids": [d.id for d in documents], "text": format_documents(documents)}

    return server


class McpRag:
    name = "mcp"

    def __init__(
        self,
        model: str,
        task: RagTask,
        settings: GenerationSettings = DEFAULT_SETTINGS,
        max_iterations: int = MAX_ITERATIONS,
    ) -> None:
        self.model = model
        self.task = task
        self.settings = settings
        self.max_iterations = max_iterations
        self.server = build_server(task.index, task.k)

    async def run(self, question: Question) -> RunResult:
        chat = ollama.AsyncClient()
        messages = [
            {"role": "system", "content": self.task.agent_instructions},
            {"role": "user", "content": question.text},
        ]
        calls: list[ToolCall] = []
        model_calls, prompt_tokens = 0, 0
        started = time.perf_counter()
        answer = f"Stopped after {self.max_iterations} model calls without an answer."

        async with Client(self.server) as client:
            schemas = [to_ollama_tool(tool) for tool in await client.list_tools()]
            for _ in range(self.max_iterations):
                response = await chat.chat(
                    model=self.model,
                    messages=messages,
                    tools=schemas,
                    options={"temperature": self.settings.temperature},
                    think=self.settings.think,
                )
                model_calls += 1
                prompt_tokens += response.get("prompt_eval_count") or 0
                message = response["message"]
                messages.append(message)
                requested = message.get("tool_calls") or []
                if not requested:
                    answer = message.get("content", "")
                    break
                for call in requested:
                    arguments = dict(call.function.arguments or {})
                    try:
                        found = (await client.call_tool(call.function.name, arguments)).data
                        text, ids = found["text"], found["ids"]
                    except ToolError as exc:
                        text, ids = f"Error: {exc}", []
                    calls.append(
                        ToolCall(
                            name=call.function.name,
                            arguments=arguments | {"k": self.task.k},
                            result=json.dumps(ids),
                        )
                    )
                    messages.append({"role": "tool", "name": call.function.name, "content": text})

        return RunResult(
            variant=self.name,
            question=question.id,
            answer=answer,
            tool_calls=calls,
            model_calls=model_calls,
            duration_seconds=time.perf_counter() - started,
            prompt_tokens=prompt_tokens,
        )
