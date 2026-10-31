"""Scoring the typed parsing case: is the answer a valid Listing, and which fields are right?

- valid: the answer validates against Listing (types and bounds);
- fields: each field of a valid answer equals the reference, prices and rating to within
  half a hundredth;
- all fields: the five at once, which is what a program reading the object relies on.

An invalid answer scores zero on every field: the program gets no object to read.
"""

from collections import defaultdict
from collections.abc import Sequence
from statistics import median, quantiles
from typing import Any

from pydantic import ValidationError

from agentic_patterns.bench import markdown_header, markdown_table, rate, seconds
from agentic_patterns.models import GenerationSettings, RunResult
from agentic_patterns.patterns.parsing.task import FIELDS, Item, Listing

TOLERANCE = 0.005
SHORT = {
    "actual_price": "actual",
    "discounted_price": "discounted",
    "discount_percent": "discount",
    "rating": "rating",
    "rating_count": "count",
}


def parse(answer: str) -> Listing | None:
    try:
        return Listing.model_validate_json(answer)
    except ValidationError:
        return None


def same(expected: float, got: float) -> bool:
    return abs(expected - got) <= TOLERANCE


def score(
    result: RunResult | None, error: str | None, item: Item, variant: str, run: int
) -> dict[str, Any]:
    """One outcome line. A run that raised, or an invalid answer, scores zero."""
    row: dict[str, Any] = {
        "run": run,
        "variant": variant,
        "item": item.id,
        "phrasing": item.phrasing,
    }
    listing = parse(result.answer) if result else None
    fields = {
        name: listing is not None and same(getattr(item.truth, name), getattr(listing, name))
        for name in FIELDS
    }
    return row | {
        "answer": result.answer if result else "",
        "seconds": round(result.duration_seconds, 3) if result else None,
        "valid": listing is not None,
        "fields": fields,
        "all_ok": all(fields.values()),
        "error": error,
    }


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One line per phrasing and variant, then one per variant over every phrasing."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    phrasings: dict[str, None] = {}
    variants: dict[str, None] = {}
    for row in rows:
        phrasings.setdefault(row["phrasing"])
        variants.setdefault(row["variant"])
        groups[(row["phrasing"], row["variant"])].append(row)
        groups[("all", row["variant"])].append(row)

    order = [(p, v) for p in [*phrasings, "all"] for v in variants if (p, v) in groups]
    return [summary_line(p, v, groups[(p, v)]) for p, v in order]


def summary_line(phrasing: str, variant: str, group: Sequence[dict[str, Any]]) -> dict[str, Any]:
    durations = [row["seconds"] for row in group if row["seconds"] is not None]
    return {
        "phrasing": phrasing,
        "variant": variant,
        "n": len(group),
        "valid": sum(row["valid"] for row in group),
        "all_ok": sum(row["all_ok"] for row in group),
        "fields": {name: sum(row["fields"][name] for row in group) for name in FIELDS},
        "errors": sum(row["error"] is not None for row in group),
        "seconds_median": median(durations) if durations else None,
        "seconds_p90": quantiles(durations, n=10)[-1] if len(durations) > 1 else None,
    }


SUMMARY_COLUMNS = (
    "phrasing",
    "variant",
    "valid",
    "all fields",
    *SHORT.values(),
    "errors",
    "s median",
    "s p90",
)
LEGEND = (
    "valid and all fields are successes/listings with their 95% Wilson interval; the five "
    "field columns count listings whose field is right. An invalid answer counts as wrong on "
    "every field. errors counts runs that raised or timed out; s median and s p90 are "
    "wall-clock seconds per listing."
)


def summary_cells(line: dict[str, Any]) -> list[str]:
    return [
        line["phrasing"],
        line["variant"],
        rate(line["valid"], line["n"]),
        rate(line["all_ok"], line["n"]),
        *(f"{line['fields'][name]}/{line['n']}" for name in FIELDS),
        str(line["errors"]),
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
    header = markdown_header(
        "Typed parsing bench",
        measured_at,
        model,
        settings,
        f"listings: {len({row['item'] for row in rows})}, runs per listing and variant: {runs}",
    )
    table = markdown_table(SUMMARY_COLUMNS, [summary_cells(line) for line in summarize(rows)])
    return "\n".join([*header, LEGEND, "", *table]) + "\n"
