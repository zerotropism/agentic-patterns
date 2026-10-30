"""Scoring the structured prompt case: did the answer follow the format, and is it right?

Two counts, kept apart:

- format: the answer is exactly one of the allowed categories, nothing before or after;
- category: the answer names exactly one allowed category, and it is the right one. An
  answer such as "Category: Home & Kitchen." fails the format and passes the category.

A model that knows the answer but does not follow the format is a different problem from
one that does not know it: the first one is fixed by parsing, the second one is not.
"""

import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from statistics import median, quantiles
from typing import Any

from agentic_patterns.bench import markdown_header, markdown_table, rate, seconds
from agentic_patterns.models import GenerationSettings, RunResult
from agentic_patterns.patterns.prompt.task import Product

OTHER = "(none or several)"


def compact(text: str) -> str:
    """Home & Kitchen, home&kitchen and HomeKitchen are the same category."""
    return re.sub(r"[^0-9a-z]", "", text.lower())


def mentioned(answer: str, categories: Sequence[str]) -> list[str]:
    found = compact(answer)
    return [category for category in categories if compact(category) in found]


def score(
    result: RunResult | None,
    error: str | None,
    product: Product,
    variant: str,
    run: int,
    categories: Sequence[str],
) -> dict[str, Any]:
    """One outcome line. A run that raised scores zero on both counts."""
    row: dict[str, Any] = {
        "run": run,
        "variant": variant,
        "product": product.id,
        "category": product.category,
    }
    if result is None:
        return row | {
            "answer": "",
            "predicted": None,
            "seconds": None,
            "format_ok": False,
            "category_ok": False,
            "error": error,
        }
    names = mentioned(result.answer, categories)
    predicted = names[0] if len(names) == 1 else None
    return row | {
        "answer": result.answer,
        "predicted": predicted,
        "seconds": round(result.duration_seconds, 3),
        "format_ok": result.answer.strip() in categories,
        "category_ok": predicted == product.category,
        "error": None,
    }


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One line per true category and variant, then one per variant over every category."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    categories: dict[str, None] = {}
    variants: dict[str, None] = {}
    for row in rows:
        categories.setdefault(row["category"])
        variants.setdefault(row["variant"])
        groups[(row["category"], row["variant"])].append(row)
        groups[("all", row["variant"])].append(row)

    order = [(c, v) for c in [*categories, "all"] for v in variants if (c, v) in groups]
    return [summary_line(c, v, groups[(c, v)]) for c, v in order]


def summary_line(category: str, variant: str, group: Sequence[dict[str, Any]]) -> dict[str, Any]:
    durations = [row["seconds"] for row in group if row["seconds"] is not None]
    return {
        "category": category,
        "variant": variant,
        "n": len(group),
        "format_ok": sum(row["format_ok"] for row in group),
        "category_ok": sum(row["category_ok"] for row in group),
        "errors": sum(row["error"] is not None for row in group),
        "seconds_median": median(durations) if durations else None,
        "seconds_p90": quantiles(durations, n=10)[-1] if len(durations) > 1 else None,
    }


SUMMARY_COLUMNS = (
    "category",
    "variant",
    "format ok",
    "category ok",
    "errors",
    "s median",
    "s p90",
)
LEGEND = (
    "Rates are successes/products with their 95% Wilson interval. format ok: the answer is "
    "exactly one allowed category. category ok: the answer names one allowed category, the "
    "right one, whatever the format. errors counts runs that raised or timed out; s median "
    "and s p90 are wall-clock seconds per product."
)


def summary_cells(line: dict[str, Any]) -> list[str]:
    return [
        line["category"],
        line["variant"],
        rate(line["format_ok"], line["n"]),
        rate(line["category_ok"], line["n"]),
        str(line["errors"]),
        seconds(line["seconds_median"]),
        seconds(line["seconds_p90"]),
    ]


def confusion(
    rows: Sequence[dict[str, Any]], variant: str, categories: Sequence[str]
) -> list[list[str]]:
    """Rows are the true categories, columns what the answer named."""
    counts = Counter(
        (row["category"], row["predicted"] or OTHER) for row in rows if row["variant"] == variant
    )
    return [
        [truth, *(str(counts[(truth, guess)]) for guess in [*categories, OTHER])]
        for truth in categories
    ]


def summary_markdown(
    rows: Sequence[dict[str, Any]],
    categories: Sequence[str],
    model: str,
    settings: GenerationSettings,
    runs: int,
    measured_at: str,
) -> str:
    header = markdown_header(
        "Structured prompt bench",
        measured_at,
        model,
        settings,
        f"products: {len({row['product'] for row in rows})}, runs per product and variant: {runs}",
    )
    table = markdown_table(SUMMARY_COLUMNS, [summary_cells(line) for line in summarize(rows)])
    parts = [*header, LEGEND, "", *table]
    for variant in dict.fromkeys(row["variant"] for row in rows):
        parts += [
            "",
            f"Confusion, `{variant}` (rows: true category, columns: category named)",
            "",
            *markdown_table(["", *categories, OTHER], confusion(rows, variant, categories)),
        ]
    return "\n".join(parts) + "\n"
