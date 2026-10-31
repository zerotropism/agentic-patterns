"""Scoring typed parsing: validity, field by field, and the whole object."""

from agentic_patterns.models import GenerationSettings, RunResult
from agentic_patterns.patterns.parsing.scoring import score, summarize, summary_markdown
from agentic_patterns.patterns.parsing.task import Item, Listing

TRUTH = Listing(
    actual_price=1099, discounted_price=399, discount_percent=64, rating=4.2, rating_count=24269
)
ITEM = Item("K", "deal", "text", TRUTH)


def answered(answer: str) -> RunResult:
    return RunResult(variant="v", question="K", answer=answer, model_calls=1, duration_seconds=0.4)


def test_a_right_answer_is_valid_and_right_everywhere() -> None:
    row = score(answered(TRUTH.model_dump_json()), None, ITEM, "raw", 0)
    assert (row["valid"], row["all_ok"]) == (True, True)


def test_prices_swapped_are_valid_but_wrong_on_two_fields() -> None:
    swapped = TRUTH.model_copy(update={"actual_price": 399, "discounted_price": 1099})
    row = score(answered(swapped.model_dump_json()), None, ITEM, "raw", 0)

    assert row["valid"] and not row["all_ok"]
    assert [name for name, ok in row["fields"].items() if not ok] == [
        "actual_price",
        "discounted_price",
    ]


def test_a_rounding_difference_within_tolerance_is_right() -> None:
    close = TRUTH.model_copy(update={"rating": 4.2001})
    assert score(answered(close.model_dump_json()), None, ITEM, "raw", 0)["all_ok"]


def test_an_invalid_answer_is_wrong_on_every_field() -> None:
    for answer in ('{"actual_price": "1,099"}', "Error: rating too high", "It costs 399."):
        row = score(answered(answer), None, ITEM, "mcp", 0)
        assert not row["valid"] and not any(row["fields"].values())


def test_a_run_that_raised_scores_zero() -> None:
    row = score(None, "TimeoutError", ITEM, "langchain", 0)
    assert (row["valid"], row["all_ok"], row["error"]) == (False, False, "TimeoutError")


def test_summary_counts_validity_and_fields_per_phrasing_then_totals() -> None:
    rows = [
        score(answered(TRUTH.model_dump_json()), None, ITEM, "raw", 0),
        score(answered("nope"), None, ITEM, "mcp", 0),
    ]
    lines = summarize(rows)

    assert [(line["phrasing"], line["variant"]) for line in lines] == [
        ("deal", "raw"),
        ("deal", "mcp"),
        ("all", "raw"),
        ("all", "mcp"),
    ]
    markdown = summary_markdown(rows, "m", GenerationSettings(), 1, "stamp")
    assert (
        "| all | raw | 1/1 (0.21-1.00) | 1/1 (0.21-1.00) | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |"
        in markdown
    )
