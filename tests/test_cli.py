"""CLI plumbing: variant resolution and question lookup. No model involved."""

import pytest

from agentic_patterns.cli import build_agent, questions_from
from agentic_patterns.patterns.agent import VARIANTS

CONFIG = {
    "model": {"name": "fake"},
    "agent": {
        "system_prompt": "p",
        "questions": {"date": "What is today's date?", "math": "What is 2 + 2?"},
    },
}


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_every_variant_can_be_built(variant: str) -> None:
    assert build_agent(variant, CONFIG).name == variant


def test_unknown_variant_lists_the_available_ones() -> None:
    with pytest.raises(ValueError, match="raw"):
        build_agent("nope", CONFIG)


def test_a_named_question_is_looked_up() -> None:
    assert questions_from(CONFIG, "date") == ["What is today's date?"]


def test_an_unknown_name_is_taken_as_the_question_itself() -> None:
    assert questions_from(CONFIG, "how tall is Everest?") == ["how tall is Everest?"]


def test_no_name_runs_every_configured_question() -> None:
    assert len(questions_from(CONFIG, None)) == 2
