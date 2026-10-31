"""What a run produces, whatever the variant that produced it."""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    """Sampling settings every variant sends to the model.

    Ollama's own defaults differ from LangChain's, so a variant that passes nothing does
    not run the same model as one that does. Every variant takes this object and sends
    both fields, which is what makes the comparison about the machinery.
    """

    temperature: float = 0.0
    think: bool = False


DEFAULT_SETTINGS = GenerationSettings()


@dataclass(frozen=True, slots=True)
class Question:
    """A question, and what a correct run looks like for the bench.

    tools None means the tools are not checked; an empty tuple means no tool may be called.
    """

    name: str
    text: str
    tools: tuple[str, ...] | None = None
    expect: tuple[str, ...] = ()

    @property
    def checked(self) -> bool:
        """Whether the bench has anything to score this question against."""
        return self.tools is not None and bool(self.expect)


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool invocation, as the variant observed it."""

    name: str
    arguments: dict
    result: str


@dataclass(frozen=True, slots=True)
class RunResult:
    """Comparable across variants: same question, same shape of answer."""

    variant: str
    question: str
    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    model_calls: int = 0
    duration_seconds: float = 0.0
    # Tokens of the last prompt sent, as Ollama counted them; only the memory case uses it
    prompt_tokens: int | None = None
