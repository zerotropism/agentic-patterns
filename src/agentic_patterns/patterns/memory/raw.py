"""Conversation memory with no framework: the history is a list, kept and sent by hand."""

import time

import ollama

from agentic_patterns.models import DEFAULT_SETTINGS, GenerationSettings, RunResult
from agentic_patterns.patterns.memory.task import Conversation


class RawMemory:
    name = "raw"

    def __init__(
        self, model: str, system: str, settings: GenerationSettings = DEFAULT_SETTINGS
    ) -> None:
        self.model = model
        self.system = system
        self.settings = settings

    async def run(self, conversation: Conversation) -> RunResult:
        """Every turn sends the whole history and appends both sides to it.

        The history is this method's local list: nothing outlives the conversation, and two
        conversations never share one.
        """
        client = ollama.AsyncClient()
        history: list[dict[str, str]] = []
        started = time.perf_counter()
        answer, prompt_tokens = "", None

        for turn in conversation.turns:
            history.append({"role": "user", "content": turn})
            response = await client.chat(
                model=self.model,
                messages=[{"role": "system", "content": self.system}, *history],
                options={"temperature": self.settings.temperature},
                think=self.settings.think,
            )
            answer = response["message"].get("content", "")
            prompt_tokens = response.get("prompt_eval_count")
            history.append({"role": "assistant", "content": answer})

        return RunResult(
            variant=self.name,
            question=conversation.id,
            answer=answer,
            model_calls=len(conversation.turns),
            duration_seconds=time.perf_counter() - started,
            prompt_tokens=prompt_tokens,
        )
