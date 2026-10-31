"""Typed parsing with no framework: Ollama constrains the output to the JSON schema, and the
answer is validated by hand against the same pydantic model.
"""

import time

import ollama
from pydantic import ValidationError

from agentic_patterns.models import DEFAULT_SETTINGS, GenerationSettings, RunResult
from agentic_patterns.patterns.parsing.task import Item, Listing, with_schema


def build_messages(instructions: str, item: Item) -> list[dict[str, str]]:
    return [{"role": "system", "content": instructions}, {"role": "user", "content": item.text}]


class RawParser:
    name = "raw"

    def __init__(
        self,
        model: str,
        instructions: str,
        settings: GenerationSettings = DEFAULT_SETTINGS,
        schema_in_prompt: bool = True,
    ) -> None:
        """schema_in_prompt=False sends the instructions alone, as measured on 30 October."""
        self.model = model
        self.instructions = with_schema(instructions) if schema_in_prompt else instructions
        self.settings = settings

    async def run(self, item: Item) -> RunResult:
        started = time.perf_counter()
        response = await ollama.AsyncClient().chat(
            model=self.model,
            messages=build_messages(self.instructions, item),
            format=Listing.model_json_schema(),
            options={"temperature": self.settings.temperature},
            think=self.settings.think,
        )
        content = response["message"].get("content", "")
        try:
            answer = Listing.model_validate_json(content).model_dump_json()
        except ValidationError:
            # Kept as the model wrote it: the bench scores it invalid
            answer = content
        return RunResult(
            variant=self.name,
            question=item.id,
            answer=answer,
            model_calls=1,
            duration_seconds=time.perf_counter() - started,
        )
