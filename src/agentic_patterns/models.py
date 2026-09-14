"""What a run produces, whatever the variant that produced it."""

from dataclasses import dataclass, field


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
