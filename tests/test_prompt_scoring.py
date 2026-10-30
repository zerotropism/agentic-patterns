"""Scoring the structured prompt case: format and category are counted apart."""

import pytest

from agentic_patterns.models import GenerationSettings, RunResult
from agentic_patterns.patterns.prompt.scoring import (
    OTHER,
    confusion,
    mentioned,
    score,
    summarize,
    summary_markdown,
)
from agentic_patterns.patterns.prompt.task import Product

CATEGORIES = ("Electronics", "Computers&Accessories", "Home&Kitchen", "OfficeProducts")
KETTLE = Product("k", "Kettle", "boils", "Home&Kitchen")


def answered(text: str) -> RunResult:
    return RunResult(variant="v", question="k", answer=text, model_calls=1, duration_seconds=0.5)


@pytest.mark.parametrize(
    ("answer", "format_ok", "category_ok"),
    [
        ("Home&Kitchen", True, True),
        ("  Home&Kitchen\n", True, True),
        ("Home & Kitchen", False, True),
        ("Category: home&kitchen.", False, True),
        ("Electronics", True, False),
        ("Electronics or Home&Kitchen", False, False),
        ("Kitchenware", False, False),
    ],
)
def test_format_and_category_are_scored_apart(answer, format_ok, category_ok) -> None:
    row = score(answered(answer), None, KETTLE, "raw", 0, CATEGORIES)
    assert (row["format_ok"], row["category_ok"]) == (format_ok, category_ok)


def test_an_answer_naming_two_categories_predicts_none() -> None:
    assert mentioned("Electronics, or Office Products", CATEGORIES) == [
        "Electronics",
        "OfficeProducts",
    ]
    assert (
        score(answered("Electronics, or Office Products"), None, KETTLE, "v", 0, CATEGORIES)[
            "predicted"
        ]
        is None
    )


def test_a_run_that_raised_scores_zero() -> None:
    row = score(None, "TimeoutError", KETTLE, "mcp", 0, CATEGORIES)
    assert (row["format_ok"], row["category_ok"], row["error"]) == (False, False, "TimeoutError")


def rows() -> list[dict]:
    lamp = Product("l", "Lamp", "light", "Electronics")
    return [
        score(answered("Home&Kitchen"), None, KETTLE, "raw", 0, CATEGORIES),
        score(answered("Electronics"), None, lamp, "raw", 0, CATEGORIES),
        score(answered("Home & Kitchen"), None, KETTLE, "mcp", 0, CATEGORIES),
        score(None, "TimeoutError", lamp, "mcp", 0, CATEGORIES),
    ]


def test_summary_has_one_line_per_category_and_variant_then_totals() -> None:
    lines = summarize(rows())

    assert [(line["category"], line["variant"]) for line in lines] == [
        ("Home&Kitchen", "raw"),
        ("Home&Kitchen", "mcp"),
        ("Electronics", "raw"),
        ("Electronics", "mcp"),
        ("all", "raw"),
        ("all", "mcp"),
    ]
    totals = {line["variant"]: line for line in lines if line["category"] == "all"}
    assert (totals["raw"]["format_ok"], totals["raw"]["category_ok"]) == (2, 2)
    assert (totals["mcp"]["format_ok"], totals["mcp"]["category_ok"], totals["mcp"]["errors"]) == (
        0,
        1,
        1,
    )


def test_confusion_counts_what_each_true_category_was_called() -> None:
    table = confusion(rows(), "mcp", CATEGORIES)
    columns = [*CATEGORIES, OTHER]
    as_dict = {row[0]: dict(zip(columns, row[1:], strict=True)) for row in table}

    assert as_dict["Home&Kitchen"]["Home&Kitchen"] == "1"
    assert as_dict["Electronics"][OTHER] == "1"


def test_the_summary_file_has_the_table_and_one_confusion_per_variant() -> None:
    markdown = summary_markdown(rows(), CATEGORIES, "m", GenerationSettings(), 1, "stamp")

    assert "- products: 2, runs per product and variant: 1" in markdown
    assert "| all | raw | 2/2 (0.34-1.00) | 2/2 (0.34-1.00) | 0 | 0.5 | 0.5 |" in markdown
    assert markdown.count("Confusion, ") == 2
