"""Terminal output for run results."""

from collections.abc import Sequence
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from agentic_patterns.bench import SUMMARY_COLUMNS, summary_cells
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


def render_table(title: str, columns: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    """A bench summary, as the markdown file has it: two label columns, then figures."""
    table = Table(title=title, title_justify="left")
    for index, column in enumerate(columns):
        table.add_column(column, justify="left" if index < 2 else "right")
    for row in rows:
        table.add_row(*(escape(cell) for cell in row))
    console.print(table)


def render_bench(lines: Sequence[dict[str, Any]]) -> None:
    render_table("Agent bench", SUMMARY_COLUMNS, [summary_cells(line) for line in lines])
