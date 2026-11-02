"""RAG with no framework: search with the question, put the results in the prompt, ask."""

import json
import time

import ollama

from agentic_patterns.models import DEFAULT_SETTINGS, GenerationSettings, RunResult, ToolCall
from agentic_patterns.patterns.rag.task import Question, RagTask, format_documents


def build_messages(instructions: str, context: str, question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": f"Product descriptions:\n\n{context}\n\nQuestion: {question}"},
    ]


def search_call(query: str, k: int, ids: list[str]) -> ToolCall:
    """A search recorded as a tool call, whoever made it, so the bench reads all three alike."""
    return ToolCall(name="search", arguments={"query": query, "k": k}, result=json.dumps(ids))


class RawRag:
    name = "raw"

    def __init__(
        self, model: str, task: RagTask, settings: GenerationSettings = DEFAULT_SETTINGS
    ) -> None:
        self.model = model
        self.task = task
        self.settings = settings

    async def run(self, question: Question) -> RunResult:
        started = time.perf_counter()
        documents = self.task.index.search(question.text, self.task.k)
        response = await ollama.AsyncClient().chat(
            model=self.model,
            messages=build_messages(
                self.task.instructions, format_documents(documents), question.text
            ),
            options={"temperature": self.settings.temperature},
            think=self.settings.think,
        )
        return RunResult(
            variant=self.name,
            question=question.id,
            answer=response["message"].get("content", ""),
            tool_calls=[search_call(question.text, self.task.k, [d.id for d in documents])],
            model_calls=1,
            duration_seconds=time.perf_counter() - started,
            prompt_tokens=response.get("prompt_eval_count"),
        )
