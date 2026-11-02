"""Scoring the RAG case: did the right document reach the model, and is the answer right?

- found: the reference document is among the results of a search the run made;
- right: the answer states the fact. A weight must give every figure with its unit
  (1 lb. 1 oz.: 1 lb and 1 oz), a country the country, USA under any of its usual names;
- right without the document: right, though no search returned the reference. The model
  knew, or guessed: either way, not retrieval.
"""

import json
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from statistics import mean, median, quantiles
from typing import Any

from agentic_patterns.bench import markdown_header, markdown_table, rate, seconds
from agentic_patterns.models import GenerationSettings, RunResult
from agentic_patterns.patterns.rag.task import Question

UNITS = {
    "lb": "lb",
    "lbs": "lb",
    "pound": "lb",
    "pounds": "lb",
    "oz": "oz",
    "ounce": "oz",
    "ounces": "oz",
    "g": "g",
    "grams": "g",
    "kg": "kg",
}
QUANTITY = re.compile(r"(\d+(?:\.\d+)?)\s*(lbs?|pounds?|oz|ounces?|grams|g|kg)\b")
USA = ("usa", "u.s.a", "u.s.", "united states", "america")


def quantities(text: str) -> Counter:
    """1 lb. 1 oz. gives {(1.0, lb): 1, (1.0, oz): 1}."""
    return Counter((float(n), UNITS[u]) for n, u in QUANTITY.findall(text.lower()))


def states(answer: str, question: Question) -> bool:
    if question.kind == "weight":
        expected = quantities(question.answer)
        return bool(expected) and not expected - quantities(answer)
    lowered = answer.lower()
    names = USA if question.answer == "USA" else (question.answer.lower(),)
    return any(re.search(rf"(?<!\w){re.escape(name)}(?!\w)", lowered) for name in names)


def found(result: RunResult, question: Question) -> bool:
    return any(question.document in json.loads(call.result) for call in result.tool_calls)


def score(
    result: RunResult | None, error: str | None, question: Question, variant: str, run: int
) -> dict[str, Any]:
    """One outcome line. A run that raised scores zero."""
    row: dict[str, Any] = {
        "run": run,
        "variant": variant,
        "question": question.id,
        "kind": question.kind,
    }
    if result is None:
        return row | {
            "answer": "",
            "tool_calls": [],
            "searches": 0,
            "found": False,
            "right": False,
            "model_calls": 0,
            "prompt_tokens": None,
            "seconds": None,
            "error": error,
        }
    return row | {
        "answer": result.answer,
        "tool_calls": [
            {"name": c.name, "arguments": c.arguments, "result": c.result}
            for c in result.tool_calls
        ],
        "searches": len(result.tool_calls),
        "found": found(result, question),
        "right": states(result.answer, question),
        "model_calls": result.model_calls,
        "prompt_tokens": result.prompt_tokens,
        "seconds": round(result.duration_seconds, 3),
        "error": None,
    }


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One line per kind of question and variant, then one per variant over every kind."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    kinds: dict[str, None] = {}
    variants: dict[str, None] = {}
    for row in rows:
        kinds.setdefault(row["kind"])
        variants.setdefault(row["variant"])
        groups[(row["kind"], row["variant"])].append(row)
        groups[("all", row["variant"])].append(row)

    order = [(k, v) for k in [*kinds, "all"] for v in variants if (k, v) in groups]
    return [summary_line(k, v, groups[(k, v)]) for k, v in order]


def summary_line(kind: str, variant: str, group: Sequence[dict[str, Any]]) -> dict[str, Any]:
    durations = [row["seconds"] for row in group if row["seconds"] is not None]
    tokens = [row["prompt_tokens"] for row in group if row["prompt_tokens"] is not None]
    return {
        "kind": kind,
        "variant": variant,
        "n": len(group),
        "found": sum(row["found"] for row in group),
        "right": sum(row["right"] for row in group),
        "right_unfound": sum(row["right"] and not row["found"] for row in group),
        "errors": sum(row["error"] is not None for row in group),
        "searches": mean(row["searches"] for row in group),
        "model_calls": mean(row["model_calls"] for row in group),
        "tokens_median": median(tokens) if tokens else None,
        "seconds_median": median(durations) if durations else None,
        "seconds_p90": quantiles(durations, n=10)[-1] if len(durations) > 1 else None,
    }


SUMMARY_COLUMNS = (
    "kind",
    "variant",
    "found",
    "right",
    "right, not found",
    "errors",
    "searches",
    "calls",
    "prompt tokens",
    "s median",
    "s p90",
)
LEGEND = (
    "found and right are questions/runs with their 95% Wilson interval: the reference "
    "document was among the search results, the answer states the fact. right, not found "
    "counts right answers without the reference document. errors counts runs that raised or "
    "timed out. searches and calls are means per question; prompt tokens is the median of "
    "all prompts of a question added up, as Ollama counted them; s median and s p90 are "
    "wall-clock seconds per question."
)


def summary_cells(line: dict[str, Any]) -> list[str]:
    tokens = line["tokens_median"]
    return [
        line["kind"],
        line["variant"],
        rate(line["found"], line["n"]),
        rate(line["right"], line["n"]),
        str(line["right_unfound"]),
        str(line["errors"]),
        f"{line['searches']:.1f}",
        f"{line['model_calls']:.1f}",
        "-" if tokens is None else f"{tokens:.0f}",
        seconds(line["seconds_median"]),
        seconds(line["seconds_p90"]),
    ]


def summary_markdown(
    rows: Sequence[dict[str, Any]],
    model: str,
    settings: GenerationSettings,
    runs: int,
    measured_at: str,
) -> str:
    questions = len({row["question"] for row in rows})
    header = markdown_header(
        "RAG bench",
        measured_at,
        model,
        settings,
        f"questions: {questions}, runs per question and variant: {runs}",
    )
    table = markdown_table(SUMMARY_COLUMNS, [summary_cells(line) for line in summarize(rows)])
    return "\n".join([*header, LEGEND, "", *table]) + "\n"
