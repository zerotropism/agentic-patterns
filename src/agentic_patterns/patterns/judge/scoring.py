"""Scoring the judge case: does the judge pick LLMBar's better output, whatever the order?

- right: the verdict names the better output, as labelled in this order;
- consistent: a pair whose two verdicts, outputs swapped, pick the same output;
- both right: a pair judged right in both orders;
- first picked: the share of verdicts naming Output (a). A judge without position bias
  picks it half the time, since each pair is shown both ways.

An answer that is neither label is invalid, and wrong.
"""

from collections import defaultdict
from collections.abc import Sequence
from statistics import median, quantiles
from typing import Any

from agentic_patterns.bench import markdown_header, markdown_table, rate, seconds
from agentic_patterns.models import GenerationSettings, RunResult
from agentic_patterns.patterns.judge.task import A, B, Judgement

SUBSETS = ("natural", "neighbor", "gptinst", "gptout", "manual")


def choice(answer: str) -> str | None:
    return answer if answer in (A, B) else None


def score(
    result: RunResult | None, error: str | None, judgement: Judgement, variant: str, run: int
) -> dict[str, Any]:
    """One outcome line. A run that raised scores zero."""
    answer = result.answer if result else ""
    picked = choice(answer)
    return {
        "run": run,
        "variant": variant,
        "judgement": judgement.id,
        "pair": judgement.pair,
        "set": judgement.set,
        "subset": judgement.subset,
        "order": judgement.order,
        "answer": answer,
        "valid": picked is not None,
        "right": picked == judgement.expected,
        "first": picked == A,
        "seconds": round(result.duration_seconds, 3) if result else None,
        "error": error,
    }


def picks_first_output(row: dict[str, Any]) -> bool | None:
    """Whether the verdict picks LLMBar's output_1, whichever order it was shown in."""
    if not row["valid"]:
        return None
    return row["first"] == (row["order"] == "ab")


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One line per subset and variant, then the adversarial subsets together, then all."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    variants: dict[str, None] = {}
    for row in rows:
        variants.setdefault(row["variant"])
        groups[(row["subset"], row["variant"])].append(row)
        if row["set"] == "adversarial":
            groups[("adversarial", row["variant"])].append(row)
        groups[("all", row["variant"])].append(row)
    names = [*(s for s in SUBSETS if any(k[0] == s for k in groups)), "adversarial", "all"]
    order = [(g, v) for g in names for v in variants if (g, v) in groups]
    return [summary_line(g, v, groups[(g, v)]) for g, v in order]


def summary_line(group: str, variant: str, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pairs[row["pair"]].append(row)
    complete = [p for p in pairs.values() if len(p) == 2]
    durations = [row["seconds"] for row in rows if row["seconds"] is not None]
    return {
        "group": group,
        "variant": variant,
        "n": len(rows),
        "right": sum(row["right"] for row in rows),
        "pairs": len(complete),
        "consistent": sum(
            picks_first_output(a) is not None and picks_first_output(a) == picks_first_output(b)
            for a, b in complete
        ),
        "both_right": sum(a["right"] and b["right"] for a, b in complete),
        "first": sum(row["first"] for row in rows),
        "invalid": sum(not row["valid"] for row in rows),
        "errors": sum(row["error"] is not None for row in rows),
        "seconds_median": median(durations) if durations else None,
        "seconds_p90": quantiles(durations, n=10)[-1] if len(durations) > 1 else None,
    }


SUMMARY_COLUMNS = (
    "subset",
    "variant",
    "right",
    "consistent",
    "both right",
    "first picked",
    "invalid",
    "errors",
    "s median",
    "s p90",
)
LEGEND = (
    "right is verdicts/judgements with its 95% Wilson interval, each pair judged twice, the "
    "outputs swapped the second time. consistent and both right are pairs: the two verdicts "
    "pick the same output, or both pick the better one. first picked is the share of verdicts "
    "naming Output (a), half for a judge without position bias. invalid counts answers that "
    "name neither; errors counts runs that raised or timed out. s median and s p90 are "
    "wall-clock seconds per verdict."
)


def summary_cells(line: dict[str, Any]) -> list[str]:
    return [
        line["group"],
        line["variant"],
        rate(line["right"], line["n"]),
        f"{line['consistent']}/{line['pairs']}",
        f"{line['both_right']}/{line['pairs']}",
        f"{line['first'] / line['n']:.2f}" if line["n"] else "-",
        str(line["invalid"]),
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
        "LLM-as-judge bench",
        measured_at,
        model,
        settings,
        f"pairs: {len({row['pair'] for row in rows})}, judged in both orders, "
        f"runs per judgement and variant: {runs}",
    )
    table = markdown_table(SUMMARY_COLUMNS, [summary_cells(line) for line in summarize(rows)])
    return "\n".join([*header, LEGEND, "", *table]) + "\n"
