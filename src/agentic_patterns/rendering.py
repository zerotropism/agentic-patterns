"""Terminal output for run results."""

from collections import Counter, defaultdict
from collections.abc import Sequence
from statistics import mean

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from agentic_patterns.models import RunResult

console = Console()


def render_result(result: RunResult) -> None:
    """Print one run: the tool calls it made, then its answer."""
    console.print(f"[bold]{result.variant}[/bold] — {escape(result.question)}")
    for call in result.tool_calls:
        console.print(
            f"  {escape(call.name)}({escape(str(call.arguments))}) -> {escape(call.result)}"
        )
    console.print(f"  [dim]{result.model_calls} model calls, {result.duration_seconds:.1f}s[/dim]")
    console.print(escape(result.answer))


def render_comparison(results: Sequence[RunResult]) -> None:
    """One row per variant and question, aggregated over the runs."""
    grouped: dict[tuple[str, str], list[RunResult]] = defaultdict(list)
    for result in results:
        grouped[(result.question, result.variant)].append(result)

    table = Table(title="Agent variants", title_justify="left")
    table.add_column("question")
    table.add_column("variant", style="bold")
    table.add_column("tools called", justify="right")
    table.add_column("model calls", justify="right")
    table.add_column("seconds", justify="right")

    for (question, variant), runs in grouped.items():
        tool_counts = Counter(len(run.tool_calls) for run in runs)
        table.add_row(
            escape(question[:40]),
            variant,
            ", ".join(f"{n}x{count}/{len(runs)}" for n, count in sorted(tool_counts.items())),
            f"{mean(run.model_calls for run in runs):.1f}",
            f"{mean(run.duration_seconds for run in runs):.1f}",
        )
    console.print(table)
