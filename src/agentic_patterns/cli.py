"""Command line entry point. One command per thing you might want to compare."""

import json
import re
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter
from rich.markup import escape

from agentic_patterns import bench as benchmark
from agentic_patterns.cases import (
    CASES,
    CaseError,
    build_agent,
    build_classifier,
    build_memory,
    build_parser,
    restore,
    select_questions,
)
from agentic_patterns.config import generation_settings, load_config, model_name
from agentic_patterns.patterns.agent import VARIANTS
from agentic_patterns.patterns.memory.task import load_task as load_memory
from agentic_patterns.patterns.parsing.task import load_task as load_parsing
from agentic_patterns.patterns.prompt.task import load_task as load_prompt
from agentic_patterns.rendering import console, render_result, render_table
from agentic_patterns.tools import TOOLS

__all__ = [
    "app",
    "build_agent",
    "build_classifier",
    "build_memory",
    "build_parser",
    "select_questions",
]

app = App(
    name="agentic-patterns",
    help="The same agentic use cases, implemented three ways.",
    version="0.1.0",
)

Variant = Annotated[str, Parameter(help=f"One of: {', '.join(VARIANTS)}.")]
ConfigPath = Annotated[str | None, Parameter(help="Path to the YAML configuration.")]


def questions_from(config: dict, name: str | None) -> list[str]:
    return [question.text for question in select_questions(config, name)]


@app.command
async def run(
    question: str,
    *,
    variant: Variant = "raw",
    config: ConfigPath = None,
) -> None:
    """Answer one question with one variant.

    The question can be a name from the configuration, or the question itself.
    """
    loaded = load_config(config)
    agent = build_agent(variant, loaded)
    for text in questions_from(loaded, question):
        render_result(await agent.run(text))


@app.command
async def classify(
    product: str | None = None,
    *,
    variant: Variant = "raw",
    config: ConfigPath = None,
) -> None:
    """Classify one product of the prompt case's sample, by id; the first one by default."""
    loaded = load_config(config)
    spec, sample = load_prompt(loaded)
    chosen = next((p for p in sample if p.id == product), None) if product else sample[0]
    if chosen is None:
        console.print(f"{escape(product or '')} is not in the sample.")
        raise SystemExit(1)
    result = await build_classifier(variant, loaded, spec).run(chosen)
    console.print(f"[bold]{result.variant}[/bold] — {escape(chosen.name[:80])}")
    console.print(f"  [dim]expected {chosen.category}, {result.duration_seconds:.1f}s[/dim]")
    console.print(escape(result.answer))


@app.command
async def extract(
    listing: str | None = None,
    *,
    variant: Variant = "raw",
    config: ConfigPath = None,
) -> None:
    """Parse one listing of the parsing case, by product id; the first one by default."""
    loaded = load_config(config)
    instructions, items = load_parsing(loaded)
    chosen = next((i for i in items if i.id == listing), None) if listing else items[0]
    if chosen is None:
        console.print(f"{escape(listing or '')} is not in the sample.")
        raise SystemExit(1)
    result = await build_parser(variant, loaded, instructions).run(chosen)
    console.print(
        f"[bold]{result.variant}[/bold] — {chosen.phrasing}, {result.duration_seconds:.1f}s"
    )
    console.print(escape(chosen.text))
    console.print(f"  [dim]expected {chosen.truth.model_dump_json()}[/dim]")
    console.print(escape(result.answer))


@app.command
async def converse(
    conversation: str | None = None,
    *,
    variant: Variant = "raw",
    config: ConfigPath = None,
) -> None:
    """Play one conversation of the memory case, by id; the first one by default."""
    loaded = load_config(config)
    system, conversations = load_memory(loaded)
    chosen = (
        next((c for c in conversations if c.id == conversation), None)
        if conversation
        else conversations[0]
    )
    if chosen is None:
        console.print(f"{escape(conversation or '')} is not in the sample.")
        raise SystemExit(1)
    result = await build_memory(variant, loaded, system).run(chosen)
    console.print(f"[bold]{result.variant}[/bold] — {chosen.id}, {result.duration_seconds:.1f}s")
    for turn in chosen.turns:
        console.print(f"  [dim]user:[/dim] {escape(turn)}")
    console.print(f"  [dim]expected {chosen.expected}, not {chosen.distractor}[/dim]")
    console.print(escape(result.answer))


Runs = Annotated[
    int | None,
    Parameter(help="How many times to run each item per variant: 20 for agent, 1 otherwise."),
]
Variants = Annotated[
    list[str] | None, Parameter(help="Variants to bench, all by default.", consume_multiple=True)
]
Case = Annotated[str, Parameter(help=f"Use case to bench: {', '.join(CASES)}.")]


@app.command
async def bench(
    *,
    case: Case = "agent",
    runs: Runs = None,
    question: str | None = None,
    variants: Variants = None,
    timeout: Annotated[float, Parameter(help="Seconds before a run counts as failed.")] = 120.0,
    output: Annotated[str, Parameter(help="Directory for outcomes and summary.")] = "evaluation",
    config: ConfigPath = None,
) -> None:
    """Run every variant of a use case, score each run, write the results.

    Outcomes are written one line per run as they come, in outcomes-<stamp>.jsonl
    (outcomes-<case>-<stamp>.jsonl for the cases other than agent); the summary, with 95%
    intervals on the rates, in the matching summary file next to it.
    """
    if case not in CASES:
        console.print(f"Unknown case '{escape(case)}'. Use one of: {', '.join(CASES)}.")
        raise SystemExit(1)
    chosen = CASES[case]
    loaded = load_config(config)
    try:
        prepared = chosen.prepare(loaded, variants, question)
    except CaseError as exc:
        console.print(escape(str(exc)))
        raise SystemExit(1) from exc
    runs = runs or chosen.runs
    model, settings = model_name(loaded), generation_settings(loaded)

    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = benchmark.timestamp()
    sink = benchmark.JsonlSink(directory / f"outcomes-{chosen.prefix}{stamp}.jsonl", model)
    total = runs * len(prepared.items) * len(prepared.variants)

    def progress(row: dict) -> None:
        sink(row)
        mark = "ok" if prepared.good(row) else row["error"] or repr(row["answer"][:40])
        console.print(
            f"[dim]{sink.count}/{total}[/dim] {row['variant']} {prepared.label(row)}: "
            f"{escape(mark)}",
            highlight=False,
        )

    console.print(f"{model}, {case} case, {runs} runs, {total} in all. Warming up.")
    try:
        rows = await benchmark.run_interleaved(
            prepared.variants,
            prepared.items,
            runs,
            timeout,
            payload=prepared.payload,
            score=prepared.score,
            sink=progress,
        )
    except RuntimeError as exc:
        console.print(escape(str(exc)))
        raise SystemExit(1) from exc

    summary = directory / f"summary-{chosen.prefix}{stamp}.md"
    summary.write_text(prepared.markdown(rows, model, settings, runs, stamp))
    render_table(prepared.title, prepared.columns, prepared.cells(rows))
    console.print(f"{sink.path}\n{summary}")


@app.command
def rescore(
    outcomes: str,
    *,
    case: Case,
    config: ConfigPath = None,
) -> None:
    """Score a stored bench run again, after a change in the scoring, without calling a model.

    Writes outcomes-<case>-<stamp>-rescored.jsonl and the matching summary next to the
    original files, which stay as they are. The agent case is refused: its expected date is
    the day of the run.
    """
    if case not in CASES or case == "agent":
        console.print("rescore applies to: " + ", ".join(c for c in CASES if c != "agent") + ".")
        raise SystemExit(1)
    source = Path(outcomes)
    stamp = re.search(r"\d{8}T\d{6}Z", source.name)
    if not source.is_file() or stamp is None:
        console.print(f"{escape(outcomes)} is not a bench outcomes file.")
        raise SystemExit(1)

    loaded = load_config(config)
    prepared = CASES[case].prepare(loaded, None, None)
    items = {prepared.item_key(item): item for item in prepared.items}
    stored = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    rows = [
        prepared.score(
            restore(row), row.get("error"), items[row[prepared.row_key]], row["variant"], row["run"]
        )
        for row in stored
    ]
    model = stored[0]["model"] if stored else model_name(loaded)
    runs = max((row["run"] for row in rows), default=0) + 1

    name = f"{CASES[case].prefix}{stamp.group()}-rescored"
    target = source.with_name(f"outcomes-{name}.jsonl")
    target.write_text("".join(json.dumps({"model": model} | row) + "\n" for row in rows))
    summary = source.with_name(f"summary-{name}.md")
    summary.write_text(
        prepared.markdown(rows, model, generation_settings(loaded), runs, stamp.group())
    )
    render_table(prepared.title, prepared.columns, prepared.cells(rows))
    console.print(f"{target}\n{summary}")


@app.command
def tools() -> None:
    """List the tools every variant shares."""
    for tool in TOOLS:
        console.print(f"[bold]{tool.__name__}[/bold] — {(tool.__doc__ or '').splitlines()[0]}")
