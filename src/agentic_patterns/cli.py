"""Command line entry point. One command per thing you might want to compare."""

from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter
from rich.markup import escape

from agentic_patterns import bench as benchmark
from agentic_patterns.config import (
    configured_questions,
    generation_settings,
    load_config,
    model_name,
)
from agentic_patterns.models import Question
from agentic_patterns.patterns.agent import VARIANTS
from agentic_patterns.protocols import Pattern
from agentic_patterns.rendering import console, render_bench, render_result
from agentic_patterns.tools import TOOLS

app = App(
    name="agentic-patterns",
    help="The same agentic use cases, implemented three ways.",
    version="0.1.0",
)

Variant = Annotated[str, Parameter(help=f"One of: {', '.join(VARIANTS)}.")]
ConfigPath = Annotated[str | None, Parameter(help="Path to the YAML configuration.")]


def build_agent(variant: str, config: dict) -> Pattern:
    """Build one variant, with the model and sampling settings every variant shares.

    Agents are reused across questions: they hold a client.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Use one of: {sorted(VARIANTS)}.")
    return VARIANTS[variant](
        model_name(config),
        config["agent"]["system_prompt"],
        settings=generation_settings(config),
    )


def select_questions(config: dict, name: str | None) -> list[Question]:
    """Named question from the configuration, all of them, or a literal question.

    A literal question has nothing to check against, so the bench scores it zero.
    """
    configured = configured_questions(config)
    if name is None:
        return list(configured.values())
    return [configured.get(name, Question(name=name, text=name))]


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


Runs = Annotated[int, Parameter(help="How many times to run each question per variant.")]
Variants = Annotated[
    list[str] | None, Parameter(help="Variants to bench, all by default.", consume_multiple=True)
]


@app.command
async def bench(
    *,
    runs: Runs = 20,
    question: str | None = None,
    variants: Variants = None,
    timeout: Annotated[float, Parameter(help="Seconds before a run counts as failed.")] = 120.0,
    output: Annotated[str, Parameter(help="Directory for outcomes and summary.")] = "evaluation",
    config: ConfigPath = None,
) -> None:
    """Run every variant on the configured questions, score each run, write the results.

    Outcomes are written one line per run as they come, in outcomes-<stamp>.jsonl; the
    summary, with 95% intervals on the rates, in summary-<stamp>.md next to it.
    """
    loaded = load_config(config)
    names = variants or list(VARIANTS)
    agents = {name: build_agent(name, loaded) for name in names}
    questions = select_questions(loaded, question)
    model, settings = model_name(loaded), generation_settings(loaded)

    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = benchmark.timestamp()
    sink = benchmark.JsonlSink(directory / f"outcomes-{stamp}.jsonl", model)
    total = runs * len(questions) * len(agents)

    def progress(row: dict) -> None:
        sink(row)
        mark = "ok" if row["tools_ok"] and row["answer_ok"] else row["error"] or "wrong"
        console.print(
            f"[dim]{sink.count}/{total}[/dim] {row['variant']} {row['question']}: {escape(mark)}",
            highlight=False,
        )

    console.print(f"{model}, {runs} runs, {total} in all. Warming up.")
    try:
        rows = await benchmark.run_bench(agents, questions, runs, timeout, progress)
    except RuntimeError as exc:
        console.print(escape(str(exc)))
        raise SystemExit(1) from exc
    summary = benchmark.write_outputs(rows, directory, stamp, model, settings, runs)

    render_bench(benchmark.summarize(rows))
    console.print(f"{sink.path}\n{summary}")


@app.command
def tools() -> None:
    """List the tools every variant shares."""
    for tool in TOOLS:
        console.print(f"[bold]{tool.__name__}[/bold] — {(tool.__doc__ or '').splitlines()[0]}")
