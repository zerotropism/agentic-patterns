"""Scoring RAG: found, right, and right without the document."""

import json

import pytest

from agentic_patterns.models import GenerationSettings, RunResult, ToolCall
from agentic_patterns.patterns.rag.scoring import (
    quantities,
    score,
    states,
    summarize,
    summary_markdown,
)
from agentic_patterns.patterns.rag.task import Question

WEIGHT = Question("weight-p0", "weight", "How much?", "p0", "1 lb. 1 oz")
ORIGIN = Question("origin-p1", "origin", "Where?", "p1", "USA")


def answered(text: str, found: list[str]) -> RunResult:
    search = ToolCall("search", {"query": "q", "k": 3}, json.dumps(found))
    return RunResult(
        variant="v",
        question="q",
        answer=text,
        tool_calls=[search],
        model_calls=1,
        duration_seconds=1.0,
        prompt_tokens=500,
    )


def test_quantities_read_figures_with_their_units() -> None:
    assert quantities("1 lb.1 oz") == quantities("one pair weighs 1 pound and 1 ounce")
    assert quantities("5.9 oz") != quantities("59 oz")


@pytest.mark.parametrize(
    ("answer", "right"),
    [
        ("They weigh 1 lb. 1 oz. per pair.", True),
        ("About 1 pound 1 ounce.", True),
        ("They weigh 1 lb.", False),
        ("They weigh 17 oz.", False),
    ],
)
def test_a_weight_needs_every_figure_and_unit(answer, right) -> None:
    assert states(answer, WEIGHT) is right


@pytest.mark.parametrize(
    ("answer", "right"),
    [
        ("It is made in the USA.", True),
        ("Made in the United States.", True),
        ("It is made in the U.S.", True),
        ("It is imported.", False),
    ],
)
def test_usa_is_recognized_under_its_usual_names(answer, right) -> None:
    assert states(answer, ORIGIN) is right


def test_found_and_right_are_scored_apart() -> None:
    row = score(answered("Made in USA.", ["p9", "p1"]), None, ORIGIN, "raw", 0)
    assert (row["found"], row["right"], row["searches"]) == (True, True, 1)

    unfound = score(answered("Made in USA.", ["p9"]), None, ORIGIN, "mcp", 0)
    assert (unfound["found"], unfound["right"]) == (False, True)


def test_a_run_that_raised_scores_zero() -> None:
    row = score(None, "TimeoutError", ORIGIN, "mcp", 0)
    assert (row["found"], row["right"], row["searches"]) == (False, False, 0)


def test_summary_counts_right_answers_without_the_document() -> None:
    rows = [
        score(answered("Made in USA.", ["p9"]), None, ORIGIN, "mcp", 0),
        score(answered("1 lb 1 oz", ["p0"]), None, WEIGHT, "mcp", 0),
    ]
    totals = summarize(rows)[-1]

    assert (totals["kind"], totals["found"], totals["right"], totals["right_unfound"]) == (
        "all",
        1,
        2,
        1,
    )
    markdown = summary_markdown(rows, "m", GenerationSettings(), 1, "stamp")
    assert "- questions: 2, runs per question and variant: 1" in markdown
