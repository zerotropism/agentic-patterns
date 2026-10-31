"""Scoring the conversation memory case: does the last answer give the fact, and only it?

- right: the answer names the fact stated at the first turn, and not the distractor;
- distractor: the answer names the distractor, the same kind of fact about someone else,
  and not the fact: the history reached the model, the wrong part of it was used.

An answer naming both, or neither, is neither right nor a distractor. Each is reported by
distance, the number of turns between the fact and the question.
"""

import re
from collections import defaultdict
from collections.abc import Sequence
from statistics import median, quantiles
from typing import Any

from agentic_patterns.bench import markdown_header, markdown_table, rate, seconds
from agentic_patterns.models import GenerationSettings, RunResult
from agentic_patterns.patterns.memory.task import Conversation


def normalize(text: str) -> str:
    """Lowercase, thousands separators and the rupee sign removed."""
    text = re.sub(r"(?<=\d)[,   ](?=\d{3}(?!\d))", "", text.lower())
    return text.replace("₹", " ")


def names(answer: str, value: str) -> bool:
    """value in answer as a whole word or number: 2,000 is not in 12,000, Pune not in
    Punekar. A word may take a plural s: "on Tuesdays" names Tuesday.
    """
    plural = "s?" if value.isalpha() else ""
    pattern = rf"(?<![\w]){re.escape(normalize(value))}{plural}(?![\w])"
    return re.search(pattern, normalize(answer)) is not None


def score(
    result: RunResult | None,
    error: str | None,
    conversation: Conversation,
    variant: str,
    run: int,
) -> dict[str, Any]:
    """One outcome line. A run that raised scores zero."""
    row: dict[str, Any] = {
        "run": run,
        "variant": variant,
        "conversation": conversation.id,
        "slot": conversation.slot,
        "distance": conversation.distance,
    }
    answer = result.answer if result else ""
    fact, other = names(answer, conversation.expected), names(answer, conversation.distractor)
    return row | {
        "answer": answer,
        "right": fact and not other,
        "distractor": other and not fact,
        "model_calls": result.model_calls if result else 0,
        "prompt_tokens": result.prompt_tokens if result else None,
        "seconds": round(result.duration_seconds, 3) if result else None,
        "error": error,
    }


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One line per distance and variant, then one per variant over every distance."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    distances: dict[str, None] = {}
    variants: dict[str, None] = {}
    for row in rows:
        key = str(row["distance"])
        distances.setdefault(key)
        variants.setdefault(row["variant"])
        groups[(key, row["variant"])].append(row)
        groups[("all", row["variant"])].append(row)

    order = [(d, v) for d in [*distances, "all"] for v in variants if (d, v) in groups]
    return [summary_line(d, v, groups[(d, v)]) for d, v in order]


def summary_line(distance: str, variant: str, group: Sequence[dict[str, Any]]) -> dict[str, Any]:
    durations = [row["seconds"] for row in group if row["seconds"] is not None]
    tokens = [row["prompt_tokens"] for row in group if row["prompt_tokens"] is not None]
    return {
        "distance": distance,
        "variant": variant,
        "n": len(group),
        "right": sum(row["right"] for row in group),
        "distractor": sum(row["distractor"] for row in group),
        "errors": sum(row["error"] is not None for row in group),
        "tokens_median": median(tokens) if tokens else None,
        "seconds_median": median(durations) if durations else None,
        "seconds_p90": quantiles(durations, n=10)[-1] if len(durations) > 1 else None,
    }


SUMMARY_COLUMNS = (
    "distance",
    "variant",
    "right",
    "distractor",
    "errors",
    "last prompt tokens",
    "s median",
    "s p90",
)
LEGEND = (
    "distance is the number of turns between the fact and the question. right and distractor "
    "are conversations/runs with their 95% Wilson interval: the last answer names the fact "
    "only, or the distractor only. errors counts runs that raised or timed out. last prompt "
    "tokens is the median size of the last prompt, as Ollama counted it; s median and s p90 "
    "are wall-clock seconds per conversation, every turn included."
)


def summary_cells(line: dict[str, Any]) -> list[str]:
    tokens = line["tokens_median"]
    return [
        line["distance"],
        line["variant"],
        rate(line["right"], line["n"]),
        rate(line["distractor"], line["n"]),
        str(line["errors"]),
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
    conversations = len({row["conversation"] for row in rows})
    header = markdown_header(
        "Conversation memory bench",
        measured_at,
        model,
        settings,
        f"conversations: {conversations}, runs per conversation and variant: {runs}",
    )
    table = markdown_table(SUMMARY_COLUMNS, [summary_cells(line) for line in summarize(rows)])
    return "\n".join([*header, LEGEND, "", *table]) + "\n"
