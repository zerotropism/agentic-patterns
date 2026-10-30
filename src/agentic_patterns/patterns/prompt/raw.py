"""Structured prompt with no framework: the messages are built by hand from the template."""

import time

import ollama

from agentic_patterns.models import DEFAULT_SETTINGS, GenerationSettings, RunResult
from agentic_patterns.patterns.prompt.task import Product, PromptSpec


def build_messages(spec: PromptSpec, product: Product) -> list[dict[str, str]]:
    """Instructions, one exchange per example, then the product to classify.

    str.format fills the placeholders. Nothing checks that every placeholder has a value
    or that every value has a placeholder: a typo in the template raises KeyError at the
    first call, or silently drops a value.
    """
    messages = [
        {"role": "user", "content": spec.instructions.format(**spec.instruction_arguments())}
    ]
    for example in spec.examples:
        messages.append(
            {
                "role": "user",
                "content": spec.user.format(name=example.name, description=example.description),
            }
        )
        messages.append({"role": "assistant", "content": example.category})
    messages.append(
        {
            "role": "user",
            "content": spec.user.format(name=product.name, description=product.description),
        }
    )
    return messages


class RawClassifier:
    name = "raw"

    def __init__(
        self, model: str, spec: PromptSpec, settings: GenerationSettings = DEFAULT_SETTINGS
    ) -> None:
        self.model = model
        self.spec = spec
        self.settings = settings

    async def run(self, product: Product) -> RunResult:
        started = time.perf_counter()
        response = await ollama.AsyncClient().chat(
            model=self.model,
            messages=build_messages(self.spec, product),
            options={"temperature": self.settings.temperature},
            think=self.settings.think,
        )
        return RunResult(
            variant=self.name,
            question=product.id,
            answer=response["message"].get("content", ""),
            model_calls=1,
            duration_seconds=time.perf_counter() - started,
        )
