"""Same prompt, built by LangChain 1.x: ChatPromptTemplate and a few-shot template."""

import time
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_ollama import ChatOllama

from agentic_patterns.models import DEFAULT_SETTINGS, GenerationSettings, RunResult
from agentic_patterns.patterns.prompt.task import Product, PromptSpec


def build_prompt(spec: PromptSpec) -> ChatPromptTemplate:
    """The template declares its variables: a missing one raises before the model is called.

    The same {placeholder} text as raw, read by LangChain's f-string templates.
    """
    example = ChatPromptTemplate.from_messages([("human", spec.user), ("ai", "{category}")])
    few_shot = FewShotChatMessagePromptTemplate(
        example_prompt=example,
        examples=[
            {"name": e.name, "description": e.description, "category": e.category}
            for e in spec.examples
        ],
    )
    prompt = ChatPromptTemplate.from_messages(
        [("human", spec.instructions), few_shot, ("human", spec.user)]
    )
    return prompt.partial(**spec.instruction_arguments())


class LangChainClassifier:
    name = "langchain"

    def __init__(
        self,
        model: str,
        spec: PromptSpec,
        settings: GenerationSettings = DEFAULT_SETTINGS,
        chat_model: Any = None,
    ) -> None:
        self.model = model
        self.spec = spec
        self.settings = settings
        self.prompt = build_prompt(spec)
        # chat_model is injected by the tests; reasoning is ChatOllama's name for think
        self.chat_model = chat_model or ChatOllama(
            model=model, temperature=settings.temperature, reasoning=settings.think
        )
        self.chain = self.prompt | self.chat_model | StrOutputParser()

    async def run(self, product: Product) -> RunResult:
        started = time.perf_counter()
        answer = await self.chain.ainvoke(
            {"name": product.name, "description": product.description}
        )
        return RunResult(
            variant=self.name,
            question=product.id,
            answer=answer,
            model_calls=1,
            duration_seconds=time.perf_counter() - started,
        )
