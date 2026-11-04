"""Judge scoring: right, consistent across orders, position bias."""

from agentic_patterns.models import GenerationSettings, RunResult
from agentic_patterns.patterns.judge.scoring import score, summarize, summary_markdown
from agentic_patterns.patterns.judge.task import A, B, judgements

PAIR = {
    "id": "p",
    "set": "adversarial",
    "subset": "manual",
    "input": "i",
    "output_1": "good",
    "output_2": "bad",
    "label": 1,
}
AB, BA = judgements(PAIR)


def verdict(answer: str) -> RunResult:
    return RunResult(variant="v", question="", answer=answer, duration_seconds=0.5)


def test_a_verdict_is_right_when_it_names_the_better_output_as_shown() -> None:
    assert score(verdict(A), None, AB, "v", 0)["right"]
    assert score(verdict(B), None, BA, "v", 0)["right"]
    assert not score(verdict(A), None, BA, "v", 0)["right"]


def test_anything_else_is_invalid_and_wrong() -> None:
    row = score(verdict("Output (a) is better."), None, AB, "v", 0)
    assert not row["valid"] and not row["right"] and not row["first"]


def test_a_judge_that_always_says_a_is_inconsistent_and_right_half_the_time() -> None:
    rows = [score(verdict(A), None, j, "v", 0) for j in (AB, BA)]
    (manual, adversarial, total) = summarize(rows)
    assert (manual["group"], adversarial["group"], total["group"]) == (
        "manual",
        "adversarial",
        "all",
    )
    assert (total["right"], total["n"], total["consistent"], total["both_right"]) == (1, 2, 0, 0)
    assert total["first"] == 2


def test_a_judge_that_follows_the_output_is_consistent() -> None:
    rows = [score(verdict(A), None, AB, "v", 0), score(verdict(B), None, BA, "v", 0)]
    total = summarize(rows)[-1]
    assert (total["consistent"], total["both_right"], total["first"]) == (1, 1, 1)


def test_consistently_wrong_is_consistent_not_right() -> None:
    rows = [score(verdict(B), None, AB, "v", 0), score(verdict(A), None, BA, "v", 0)]
    total = summarize(rows)[-1]
    assert (total["consistent"], total["right"]) == (1, 0)


def test_an_invalid_verdict_breaks_consistency() -> None:
    rows = [score(verdict(A), None, AB, "v", 0), score(verdict("?"), None, BA, "v", 0)]
    assert summarize(rows)[-1]["consistent"] == 0


def test_the_summary_counts_pairs() -> None:
    rows = [score(verdict(A), None, AB, "v", 0), score(None, "TimeoutError", BA, "v", 0)]
    text = summary_markdown(rows, "m", GenerationSettings(), 1, "20261103T120000Z")
    assert "pairs: 1, judged in both orders" in text and "| all | v |" in text
