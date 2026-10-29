"""Many runs of every variant on every question, scored, with confidence intervals.

A run is scored on two things, independently:

- tools: every tool the question needs was called, and returned without error at least
  once; for a question that needs no tool, no tool was called at all;
- answer: every expected value appears in the final answer.

A run can call the right tools and still answer wrong, or skip the tools and happen to
be right. The second case is the one the system prompt forbids, so both are reported.
Tool calls that returned an error are counted apart: a run can recover from one.
"""

import asyncio
import json
import math
import re
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import date
from pathlib import Path
from statistics import mean, median, quantiles
from typing import Any

from agentic_patterns.models import GenerationSettings, Question, RunResult
from agentic_patterns.protocols import Pattern
from agentic_patterns.tools import calculate

TODAY = "{today}"
Z_95 = 1.96

# A single separator between digits, followed by exactly three digits: 1,018,022 or 1 095
_THOUSANDS = re.compile(r"(?<=\d)[,   ](?=\d{3}(?!\d))")
_ORDINAL = re.compile(r"(?<=\d)(st|nd|rd|th)\b")


def normalize(text: str) -> str:
    """Lowercase, thousands separators and ordinals removed, punctuation to spaces."""
    text = _THOUSANDS.sub("", text.lower())
    text = _ORDINAL.sub("", text)
    text = re.sub(r"[,*]", " ", text)
    return " ".join(text.split())


def date_forms(day: date) -> list[str]:
    """The ways a model writes a date: 2026-10-20, october 20 2026, 20 (of) october 2026,
    10/20/2026 or 20/10/2026. A weekday in front changes nothing: the match is a substring.
    """
    month = day.strftime("%B").lower()
    return [
        day.isoformat(),
        f"{month} {day.day} {day.year}",
        f"{day.day} {month} {day.year}",
        f"{day.day} of {month} {day.year}",
        f"{day.month}/{day.day}/{day.year}",
        f"{day.day}/{day.month}/{day.year}",
    ]


def computed(expression: str, today: date) -> str:
    """An expected value that depends on the day: '=2100 - {year}' gives 74 in 2026.

    Evaluated by the same arithmetic the calculate tool uses, so nothing else runs.
    """
    value = calculate(expression.format(year=today.year, month=today.month, day=today.day))
    return value.removesuffix(".0")


def alternatives(expected: str, today: date) -> list[str]:
    """What counts as the expected value in an answer."""
    if expected == TODAY:
        return date_forms(today)
    if expected.startswith("="):
        return [computed(expected[1:], today)]
    return [normalize(expected)]


def contains(answer: str, value: str) -> bool:
    """value in answer, without matching 1095 inside 10950."""
    return re.search(rf"(?<!\d){re.escape(value)}(?!\d)", answer) is not None


def answer_ok(answer: str, question: Question, today: date) -> bool:
    normalized = normalize(answer)
    return all(
        any(contains(normalized, value) for value in alternatives(expected, today))
        for expected in question.expect
    )


def failed(result_text: str) -> bool:
    """raw and mcp report a failing tool to the model as text starting with Error."""
    return result_text.startswith("Error")


def tools_ok(result: RunResult, question: Question) -> bool:
    if question.tools is None:
        return True
    if not question.tools:
        return not result.tool_calls
    succeeded = {call.name for call in result.tool_calls if not failed(call.result)}
    return set(question.tools) <= succeeded


def wilson(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """95% interval for a rate. Unlike the normal approximation, sound at 0/n and n/n."""
    if n == 0:
        return 0.0, 1.0
    rate = successes / n
    centre = (rate + z * z / (2 * n)) / (1 + z * z / n)
    margin = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - margin), min(1.0, centre + margin)


def score(
    result: RunResult | None,
    error: str | None,
    question: Question,
    variant: str,
    run: int,
    today: date,
) -> dict[str, Any]:
    """One outcome line. A run that raised scores zero on both counts."""
    row: dict[str, Any] = {"run": run, "variant": variant, "question": question.name}
    if result is None:
        return row | {
            "tools": [],
            "tool_calls": [],
            "tool_errors": 0,
            "answer": "",
            "model_calls": 0,
            "seconds": None,
            "tools_ok": False,
            "answer_ok": False,
            "error": error,
        }
    return row | {
        "tools": [call.name for call in result.tool_calls],
        "tool_calls": [asdict(call) for call in result.tool_calls],
        "tool_errors": sum(failed(call.result) for call in result.tool_calls),
        "answer": result.answer,
        "model_calls": result.model_calls,
        "seconds": round(result.duration_seconds, 3),
        "tools_ok": tools_ok(result, question),
        "answer_ok": answer_ok(result.answer, question, today),
        "error": None,
    }


async def attempt(agent: Pattern, text: str, timeout: float) -> tuple[RunResult | None, str | None]:
    """Run once. Anything raised is recorded as the outcome, never ends the bench."""
    try:
        return await asyncio.wait_for(agent.run(text), timeout), None
    except Exception as exc:
        # TimeoutError and httpx timeouts carry an empty message; the type is the information
        return None, f"{type(exc).__name__}: {exc}".rstrip(": ")


def rotated(names: Sequence[str], run: int) -> list[str]:
    """Shift the variant order each run, so none always goes first after a cold prompt."""
    shift = run % len(names)
    return [*names[shift:], *names[:shift]]


async def run_bench(
    agents: Mapping[str, Pattern],
    questions: Sequence[Question],
    runs: int,
    timeout: float,
    sink: Callable[[dict[str, Any]], None],
    today: date | None = None,
    warmup: bool = True,
) -> list[dict[str, Any]]:
    """Every question, every variant, runs times, interleaved. Each outcome goes to sink
    as soon as it exists, so an interrupted bench keeps what it measured.
    """
    today = today or date.today()
    names = list(agents)
    rows: list[dict[str, Any]] = []

    if warmup and questions:
        # Loads the model and fills the prompt cache; not measured
        errors = [(await attempt(agents[name], questions[0].text, timeout))[1] for name in names]
        if all(errors):
            raise RuntimeError(f"Every warm-up run failed, nothing to measure: {errors[0]}")

    for run in range(runs):
        for question in questions:
            for name in rotated(names, run):
                result, error = await attempt(agents[name], question.text, timeout)
                row = score(result, error, question, name, run, today)
                rows.append(row)
                sink(row)
    return rows


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One line per question and variant, then one per variant over every question.

    Questions and variants keep the order they were run in, whatever the rotation.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    questions: dict[str, None] = {}
    variants: dict[str, None] = {}
    for row in rows:
        questions.setdefault(row["question"])
        variants.setdefault(row["variant"])
        groups[(row["question"], row["variant"])].append(row)
        groups[("all", row["variant"])].append(row)

    order = [(q, v) for q in [*questions, "all"] for v in variants if (q, v) in groups]
    return [summary_line(q, v, groups[(q, v)]) for q, v in order]


def summary_line(question: str, variant: str, group: Sequence[dict[str, Any]]) -> dict[str, Any]:
    durations = [row["seconds"] for row in group if row["seconds"] is not None]
    return {
        "question": question,
        "variant": variant,
        "n": len(group),
        "tools_ok": sum(row["tools_ok"] for row in group),
        "answer_ok": sum(row["answer_ok"] for row in group),
        "errors": sum(row["error"] is not None for row in group),
        "tool_errors": sum(row["tool_errors"] for row in group),
        "model_calls": mean(row["model_calls"] for row in group),
        "seconds_median": median(durations) if durations else None,
        "seconds_p90": quantiles(durations, n=10)[-1] if len(durations) > 1 else None,
    }


def rate(successes: int, n: int) -> str:
    low, high = wilson(successes, n)
    return f"{successes}/{n} ({low:.2f}-{high:.2f})"


def seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


SUMMARY_COLUMNS = (
    "question",
    "variant",
    "tools ok",
    "answer ok",
    "errors",
    "tool errors",
    "calls",
    "s median",
    "s p90",
)
LEGEND = (
    "Rates are successes/runs with their 95% Wilson interval. errors counts runs that raised "
    "or timed out; tool errors counts tool calls that returned an error, over all runs. calls "
    "is the mean number of model calls per run; s median and s p90 are wall-clock seconds "
    "per run."
)


def summary_cells(line: dict[str, Any]) -> list[str]:
    return [
        line["question"],
        line["variant"],
        rate(line["tools_ok"], line["n"]),
        rate(line["answer_ok"], line["n"]),
        str(line["errors"]),
        str(line["tool_errors"]),
        f"{line['model_calls']:.1f}",
        seconds(line["seconds_median"]),
        seconds(line["seconds_p90"]),
    ]


def summary_markdown(
    lines: Sequence[dict[str, Any]],
    model: str,
    settings: GenerationSettings,
    runs: int,
    measured_at: str,
) -> str:
    header = [
        "# Agent bench",
        "",
        f"- measured: {measured_at}",
        f"- model: `{model}`, temperature {settings.temperature}, think {settings.think}",
        f"- runs per question and variant: {runs}",
        "",
        LEGEND,
        "",
        "| " + " | ".join(SUMMARY_COLUMNS) + " |",
        "|" + "---|" * len(SUMMARY_COLUMNS),
    ]
    body = ["| " + " | ".join(summary_cells(line)) + " |" for line in lines]
    return "\n".join(header + body) + "\n"


def write_outputs(
    rows: Sequence[dict[str, Any]],
    output: Path,
    stamp: str,
    model: str,
    settings: GenerationSettings,
    runs: int,
) -> Path:
    """The summary next to the outcomes it was computed from."""
    path = output / f"summary-{stamp}.md"
    path.write_text(summary_markdown(summarize(rows), model, settings, runs, stamp))
    return path


class JsonlSink:
    """Appends one outcome per line and flushes, so a crash loses at most one run."""

    def __init__(self, path: Path, model: str) -> None:
        self.path = path
        self.model = model
        self.count = 0

    def __call__(self, row: dict[str, Any]) -> None:
        with self.path.open("a") as handle:
            handle.write(json.dumps({"model": self.model} | row, ensure_ascii=False) + "\n")
        self.count += 1


def timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
