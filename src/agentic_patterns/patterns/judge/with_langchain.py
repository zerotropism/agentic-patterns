"""The same judge built with openevals, LangChain's library of evaluators.

create_async_llm_as_judge takes the rubric as a prompt template and a chat model, and asks
for structured output. Given the Verdict schema, it sends the request raw sends: ChatOllama's
with_structured_output passes the schema as Ollama's format=. Its default would ask for a
score with a written reasoning first; LLMBar's Vanilla rubric asks for no explanation.
"""

import time
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from openevals.llm import create_async_llm_as_judge

from agentic_patterns.models import DEFAULT_SETTINGS, GenerationSettings, RunResult
from agentic_patterns.patterns.judge.task import Judgement, Verdict


class LangChainJudge:
    name = "langchain"

    def __init__(
        self,
        model: str,
        prompts: dict[str, str],
        settings: GenerationSettings = DEFAULT_SETTINGS,
        chat_model: Any = None,
    ) -> None:
        self.model = model
        self.settings = settings
        # chat_model is injected by the tests; reasoning is ChatOllama's name for think
        self.chat_model = chat_model or ChatOllama(
            model=model, temperature=settings.temperature, reasoning=settings.think
        )
        self.evaluator = create_async_llm_as_judge(
            prompt=ChatPromptTemplate.from_messages([("user", prompts["json"])]),
            judge=self.chat_model,
            output_schema=Verdict.model_json_schema(),
        )

    async def run(self, judgement: Judgement) -> RunResult:
        started = time.perf_counter()
        try:
            verdict = await self.evaluator(
                instruction=judgement.instruction,
                output_a=judgement.output_a,
                output_b=judgement.output_b,
            )
            answer = str(verdict.get("better", verdict))
        except ValueError as exc:
            # An answer the schema refuses: the bench scores it invalid
            answer = f"Error: {exc}"
        return RunResult(
            variant=self.name,
            question=judgement.id,
            answer=answer,
            model_calls=1,
            duration_seconds=time.perf_counter() - started,
        )
