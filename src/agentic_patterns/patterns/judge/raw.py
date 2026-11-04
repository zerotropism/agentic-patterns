"""The judge with no framework: the rubric is formatted by hand, Ollama constrains the
answer to the Verdict schema, and the answer is validated against it.
"""

import time

import ollama
from pydantic import ValidationError

from agentic_patterns.models import DEFAULT_SETTINGS, GenerationSettings, RunResult
from agentic_patterns.patterns.judge.task import Judgement, Verdict, render


def build_messages(prompt: str, judgement: Judgement) -> list[dict[str, str]]:
    """One user message: LLMBar puts the first sentence in a system message, but every
    variant must send the same roles, and an MCP prompt has no system role.
    """
    return [{"role": "user", "content": render(prompt, judgement)}]


class RawJudge:
    name = "raw"

    def __init__(
        self, model: str, prompts: dict[str, str], settings: GenerationSettings = DEFAULT_SETTINGS
    ) -> None:
        self.model = model
        self.prompt = prompts["json"]
        self.settings = settings

    async def run(self, judgement: Judgement) -> RunResult:
        started = time.perf_counter()
        response = await ollama.AsyncClient().chat(
            model=self.model,
            messages=build_messages(self.prompt, judgement),
            format=Verdict.model_json_schema(),
            options={"temperature": self.settings.temperature},
            think=self.settings.think,
        )
        content = response["message"].get("content", "")
        try:
            answer = Verdict.model_validate_json(content).better
        except ValidationError:
            # Kept as the model wrote it: the bench scores it invalid
            answer = content
        return RunResult(
            variant=self.name,
            question=judgement.id,
            answer=answer,
            model_calls=1,
            duration_seconds=time.perf_counter() - started,
        )
