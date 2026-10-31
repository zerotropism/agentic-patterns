"""Scoring conversation memory: the fact, the distractor, both or neither."""

import pytest

from agentic_patterns.models import GenerationSettings, RunResult
from agentic_patterns.patterns.memory.scoring import names, score, summarize, summary_markdown
from agentic_patterns.patterns.memory.task import Conversation

BUDGET = Conversation("budget-3-0", "budget", 3, ("…",), expected="2,000", distractor="12,000")
CITY = Conversation("city-1-0", "city", 1, ("…",), expected="Pune", distractor="Surat")


def answered(text: str, tokens: int = 120) -> RunResult:
    return RunResult(
        variant="v",
        question="c",
        answer=text,
        model_calls=3,
        duration_seconds=2.0,
        prompt_tokens=tokens,
    )


@pytest.mark.parametrize(
    ("answer", "found"),
    [
        ("Your budget is ₹2,000.", True),
        ("Your budget is 2000 rupees.", True),
        ("Your budget is Rs 2 000.", True),
        ("Your budget is ₹12,000.", False),
        ("Your budget is ₹2,000.50.", True),
    ],
)
def test_amounts_are_found_whatever_their_format(answer, found) -> None:
    assert names(answer, "2,000") is found


def test_names_are_whole_words() -> None:
    assert names("You live in Pune.", "Pune")
    assert not names("You live in Punekar Nagar.", "Pune")


def test_a_word_may_take_a_plural_s() -> None:
    """Measured on 31 October: "deliveries on Tuesdays" was scored wrong before this."""
    assert names("You can receive deliveries on Tuesdays.", "Tuesday")
    assert not names("You can receive deliveries on Tuesdayish.", "Tuesday")


@pytest.mark.parametrize(
    ("answer", "right", "distractor"),
    [
        ("You live in Pune.", True, False),
        ("You live in Surat.", False, True),
        ("You live in Pune; your sister in Surat.", False, False),
        ("I don't know where you live.", False, False),
    ],
)
def test_right_and_distractor_exclude_each_other(answer, right, distractor) -> None:
    row = score(answered(answer), None, CITY, "raw", 0)
    assert (row["right"], row["distractor"]) == (right, distractor)


def test_a_run_that_raised_scores_zero() -> None:
    row = score(None, "TimeoutError", BUDGET, "mcp", 0)
    assert (row["right"], row["distractor"], row["prompt_tokens"]) == (False, False, None)


def test_summary_has_one_line_per_distance_and_variant_then_totals() -> None:
    rows = [
        score(answered("You live in Pune."), None, CITY, "raw", 0),
        score(answered("₹12,000", tokens=300), None, BUDGET, "raw", 0),
        score(answered("You live in Pune."), None, CITY, "mcp", 0),
    ]
    lines = summarize(rows)

    assert [(line["distance"], line["variant"]) for line in lines] == [
        ("1", "raw"),
        ("1", "mcp"),
        ("3", "raw"),
        ("all", "raw"),
        ("all", "mcp"),
    ]
    markdown = summary_markdown(rows, "m", GenerationSettings(), 1, "stamp")
    assert "| all | raw | 1/2 (0.09-0.91) | 1/2 (0.09-0.91) | 0 | 210 | 2.0 | 2.0 |" in markdown
