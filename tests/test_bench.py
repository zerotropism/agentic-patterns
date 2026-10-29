"""Bench scoring, intervals and the run loop, with fake agents. No Ollama."""

import asyncio
import json
from datetime import date

import pytest

from agentic_patterns.bench import (
    JsonlSink,
    answer_ok,
    rotated,
    run_bench,
    score,
    summarize,
    summary_markdown,
    tools_ok,
    wilson,
)
from agentic_patterns.models import GenerationSettings, Question, RunResult, ToolCall

DAY = date(2026, 10, 20)
DATE = Question("date", "date?", tools=("today",), expect=("{today}",))
MATH = Question("math", "math?", tools=("calculate",), expect=("1018022",))
COMBINED = Question("combined", "both?", tools=("today", "calculate"), expect=("{today}", "1095"))
CHAINED = Question("chained", "2100?", tools=("today", "calculate"), expect=("=2100 - {year}",))
NO_TOOL = Question("no_tool", "capital?", tools=(), expect=("Paris",))


@pytest.mark.parametrize(
    "answer",
    [
        "Today's date is 2026-10-20.",
        "Today is Tuesday, October 20, 2026.",
        "It is the 20th of October 2026",
        "October 20th, 2026",
        "**10/20/2026**",
    ],
)
def test_usual_ways_of_writing_the_date_are_accepted(answer: str) -> None:
    assert answer_ok(answer, DATE, DAY)


@pytest.mark.parametrize("answer", ["Today is October 21, 2026.", "2025-10-20", "I can't know."])
def test_another_date_is_refused(answer: str) -> None:
    assert not answer_ok(answer, DATE, DAY)


@pytest.mark.parametrize("answer", ["1018022", "The result is 1,018,022.", "= 1 018 022"])
def test_thousands_separators_are_ignored(answer: str) -> None:
    assert answer_ok(answer, MATH, DAY)


def test_a_number_inside_a_longer_one_does_not_count() -> None:
    assert not answer_ok("2026-10-20 and 10950", COMBINED, DAY)


def test_every_expected_value_is_needed() -> None:
    assert answer_ok("2026-10-20, and 365 * 3 = 1,095", COMBINED, DAY)
    assert not answer_ok("365 * 3 = 1,095", COMBINED, DAY)


def result(*calls: tuple[str, str], answer: str = "") -> RunResult:
    return RunResult(
        variant="v",
        question="q",
        answer=answer,
        tool_calls=[ToolCall(name, {}, output) for name, output in calls],
        model_calls=len(calls) + 1,
        duration_seconds=1.0,
    )


def test_tools_ok_needs_every_tool_to_succeed() -> None:
    assert tools_ok(result(("today", "2026-10-20"), ("calculate", "1095")), COMBINED)
    assert not tools_ok(result(("today", "2026-10-20")), COMBINED)
    assert not tools_ok(result(("today", "x"), ("calculate", "Error: bad")), COMBINED)


def test_a_computed_value_depends_on_the_day() -> None:
    assert answer_ok("74 years are left.", CHAINED, DAY)
    assert not answer_ok("76 years are left.", CHAINED, DAY)
    assert answer_ok("73 years", CHAINED, date(2027, 1, 1))


def test_a_question_needing_no_tool_fails_if_one_is_called() -> None:
    assert tools_ok(result(answer="Paris"), NO_TOOL)
    assert not tools_ok(result(("today", "2026-10-20"), answer="Paris"), NO_TOOL)


def test_an_unchecked_question_accepts_any_tools() -> None:
    assert tools_ok(result(("today", "x")), Question("q", "q"))


def test_a_tool_error_followed_by_a_success_still_counts() -> None:
    """The recovery question: the first attempt fails, the second one is right."""
    recovery = Question("recovery", "?", tools=("calculate",), expect=("1036",))
    run = result(("calculate", "Error: Unsupported expression element"), ("calculate", "1036.0"))
    assert tools_ok(run, recovery)
    assert score(run, None, recovery, "v", 0, DAY)["tool_errors"] == 1


def test_wilson_interval_is_sound_at_the_extremes() -> None:
    low, high = wilson(20, 20)
    assert high == 1.0 and 0.8 < low < 0.85
    assert wilson(0, 20)[0] == 0.0
    assert wilson(0, 0) == (0.0, 1.0)


def test_the_variant_order_rotates() -> None:
    assert [rotated(["a", "b", "c"], run)[0] for run in range(4)] == ["a", "b", "c", "a"]


class FakeAgent:
    def __init__(self, name: str, answer: str = "1018022", fail: bool = False) -> None:
        self.name = name
        self.answer = answer
        self.fail = fail
        self.calls = 0

    async def run(self, question: str) -> RunResult:
        self.calls += 1
        if self.fail:
            raise ConnectionError("Ollama is down")
        return RunResult(
            variant=self.name,
            question=question,
            answer=self.answer,
            tool_calls=[ToolCall("calculate", {"expression": "x"}, "1018022")],
            model_calls=2,
            duration_seconds=0.5,
        )


class SlowAgent(FakeAgent):
    async def run(self, question: str) -> RunResult:
        await asyncio.sleep(1)
        return await super().run(question)


async def test_every_run_is_scored_and_sent_to_the_sink() -> None:
    agents = {"good": FakeAgent("good"), "wrong": FakeAgent("wrong", answer="42")}
    seen = []
    rows = await run_bench(agents, [MATH], runs=3, timeout=5, sink=seen.append, today=DAY)

    assert rows == seen and len(rows) == 6
    assert {row["variant"]: row["answer_ok"] for row in rows} == {"good": True, "wrong": False}
    # one warm-up run each, not scored
    assert agents["good"].calls == 4


async def test_a_failing_run_is_recorded_not_raised() -> None:
    rows = await run_bench(
        {"down": FakeAgent("down", fail=True)}, [MATH], 2, 5, lambda _: None, DAY, warmup=False
    )
    assert [row["error"] for row in rows] == ["ConnectionError: Ollama is down"] * 2
    assert not any(row["tools_ok"] or row["answer_ok"] for row in rows)


async def test_a_bench_whose_warm_up_all_fails_stops_before_measuring() -> None:
    agent = FakeAgent("down", fail=True)
    with pytest.raises(RuntimeError, match="Ollama is down"):
        await run_bench({"down": agent}, [MATH], 20, 5, lambda _: None, DAY)
    assert agent.calls == 1


async def test_a_run_over_the_timeout_counts_as_failed() -> None:
    rows = await run_bench(
        {"slow": SlowAgent("slow")}, [MATH], 1, 0.05, lambda _: None, DAY, warmup=False
    )
    assert rows[0]["error"] == "TimeoutError"


async def test_summary_has_one_line_per_question_and_variant_then_totals() -> None:
    agents = {"a": FakeAgent("a"), "b": FakeAgent("b", answer="42")}
    rows = await run_bench(agents, [MATH, DATE], 2, 5, lambda _: None, DAY, warmup=False)
    lines = summarize(rows)

    assert [(line["question"], line["variant"]) for line in lines] == [
        ("math", "a"),
        ("math", "b"),
        ("date", "a"),
        ("date", "b"),
        ("all", "a"),
        ("all", "b"),
    ]
    totals = {line["variant"]: line for line in lines if line["question"] == "all"}
    assert (totals["a"]["n"], totals["a"]["answer_ok"], totals["b"]["answer_ok"]) == (4, 2, 0)

    markdown = summary_markdown(lines, "m", GenerationSettings(), 2, "stamp")
    assert "| math | a | 2/2 (0.34-1.00) | 2/2 (0.34-1.00) | 0 | 0 | 2.0 | 0.5 | 0.5 |" in markdown


def test_the_sink_appends_one_json_line_per_run(tmp_path) -> None:
    sink = JsonlSink(tmp_path / "outcomes.jsonl", "m")
    sink({"run": 0})
    sink({"run": 1})

    lines = (tmp_path / "outcomes.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in lines] == [
        {"model": "m", "run": 0},
        {"model": "m", "run": 1},
    ]
