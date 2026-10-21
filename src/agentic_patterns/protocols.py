"""What every variant of every use case implements."""

from typing import Protocol, runtime_checkable

from agentic_patterns.models import RunResult


@runtime_checkable
class Pattern(Protocol):
    """One way of answering a question: a name to tell it apart, and an async run.

    Variants share nothing else. The CLI and the bench only rely on this, so a new
    variant needs no change outside its own module and the registry that lists it.
    """

    name: str

    async def run(self, question: str) -> RunResult: ...
