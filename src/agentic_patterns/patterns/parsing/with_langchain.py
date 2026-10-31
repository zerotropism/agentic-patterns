"""Same parsing, built by LangChain 1.x: with_structured_output on the pydantic model.

ChatOllama's default method is json_schema: the same constrained output as raw, the schema
written by the framework from the same model, and the validation done for you. Like raw's
format=, it does not show the schema to the model: that is left to the prompt.
"""

import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from agentic_patterns.models import DEFAULT_SETTINGS, GenerationSettings, RunResult
from agentic_patterns.patterns.parsing.task import Item, Listing, with_schema


class LangChainParser:
    name = "langchain"

    def __init__(
        self,
        model: str,
        instructions: str,
        settings: GenerationSettings = DEFAULT_SETTINGS,
        chat_model: Any = None,
        schema_in_prompt: bool = True,
    ) -> None:
        """schema_in_prompt=False sends the instructions alone, as measured on 30 October."""
        self.model = model
        self.instructions = with_schema(instructions) if schema_in_prompt else instructions
        self.settings = settings
        # chat_model is injected by the tests; reasoning is ChatOllama's name for think
        self.chat_model = chat_model or ChatOllama(
            model=model, temperature=settings.temperature, reasoning=settings.think
        )
        # include_raw: a failed parse comes back as data instead of an exception
        self.structured = self.chat_model.with_structured_output(Listing, include_raw=True)

    async def run(self, item: Item) -> RunResult:
        started = time.perf_counter()
        output = await self.structured.ainvoke(
            [SystemMessage(self.instructions), HumanMessage(item.text)]
        )
        parsed = output["parsed"]
        answer = parsed.model_dump_json() if parsed else str(output["raw"].content)
        return RunResult(
            variant=self.name,
            question=item.id,
            answer=answer,
            model_calls=1,
            duration_seconds=time.perf_counter() - started,
        )
