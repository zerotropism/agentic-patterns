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
from agentic_patterns.patterns.prompt import VARIANTS as PROMPT_VARIANTS
from agentic_patterns.patterns.prompt import scoring as prompt_scoring
from agentic_patterns.patterns.prompt.task import PromptSpec, load_task
from agentic_patterns.protocols import Pattern
from agentic_patterns.rendering import console, render_bench, render_result, render_table
from agentic_patterns.tools import TOOLS

app = App(
    name="agentic-patterns",
    help="The same agentic use cases, implemented three ways.",
    version="0.1.0",
)

CASES = ("agent", "prompt")
# The agent questions are few and each one is worth repeating; the prompt case has a
# hundred products, and repeating them at temperature 0 adds little
DEFAULT_RUNS = {"agent": 20, "prompt": 1}

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


def build_classifier(variant: str, config: dict, spec: PromptSpec) -> Pattern:
    """One variant of the structured prompt case, with the shared model and settings."""
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Use one of: {sorted(PROMPT_VARIANTS)}.")
    return PROMPT_VARIANTS[variant](model_name(config), spec, settings=generation_settings(config))


def select_questions(config: dict, name: str | None) -> list[Question]:
    """Named question from the configuration, all of them, or a literal question.

    A literal question has nothing to check against: run answers it, bench refuses it.
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


@app.command
async def classify(
    product: str | None = None,
    *,
    variant: Variant = "raw",
    config: ConfigPath = None,
) -> None:
    """Classify one product of the prompt case's sample, by id; the first one by default."""
    loaded = load_config(config)
    spec, sample = load_task(loaded)
    chosen = next((p for p in sample if p.id == product), None) if product else sample[0]
    if chosen is None:
        console.print(f"{escape(product or '')} is not in the sample.")
        raise SystemExit(1)
    result = await build_classifier(variant, loaded, spec).run(chosen)
    console.print(f"[bold]{result.variant}[/bold] — {escape(chosen.name[:80])}")
    console.print(f"  [dim]expected {chosen.category}, {result.duration_seconds:.1f}s[/dim]")
    console.print(escape(result.answer))


Runs = Annotated[
    int | None,
    Parameter(help="How many times to run each item per variant: 20 for agent, 1 for prompt."),
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
    (outcomes-prompt-<stamp>.jsonl for the prompt case); the summary, with 95% intervals
    on the rates, in the matching summary file next to it.
    """
    if case not in CASES:
        console.print(f"Unknown case '{escape(case)}'. Use one of: {', '.join(CASES)}.")
        raise SystemExit(1)
    loaded = load_config(config)
    runs = runs or DEFAULT_RUNS[case]
    model, settings = model_name(loaded), generation_settings(loaded)

    if case == "agent":
        items = select_questions(loaded, question)
        unchecked = [q.name for q in items if not q.checked]
        if unchecked:
            console.print(f"No tools or expect configured for: {escape(', '.join(unchecked))}.")
            raise SystemExit(1)
        variants_built = {n: build_agent(n, loaded) for n in variants or list(VARIANTS)}
        prefix = ""
    else:
        if question:
            console.print("--question applies to the agent case only.")
            raise SystemExit(1)
        spec, items = load_task(loaded)
        variants_built = {
            n: build_classifier(n, loaded, spec) for n in variants or list(PROMPT_VARIANTS)
        }
        prefix = "prompt-"

    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = benchmark.timestamp()
    sink = benchmark.JsonlSink(directory / f"outcomes-{prefix}{stamp}.jsonl", model)
    total = runs * len(items) * len(variants_built)

    def progress(row: dict) -> None:
        sink(row)
        if case == "agent":
            label, good = row["question"], row["tools_ok"] and row["answer_ok"]
        else:
            label, good = row["product"], row["format_ok"] and row["category_ok"]
        mark = "ok" if good else row["error"] or repr(row["answer"][:40])
        console.print(
            f"[dim]{sink.count}/{total}[/dim] {row['variant']} {label}: {escape(mark)}",
            highlight=False,
        )

    console.print(f"{model}, {case} case, {runs} runs, {total} in all. Warming up.")
    try:
        if case == "agent":
            rows = await benchmark.run_bench(variants_built, items, runs, timeout, progress)
        else:
            rows = await benchmark.run_interleaved(
                variants_built,
                items,
                runs,
                timeout,
                payload=lambda product: product,
                score=lambda result, error, product, name, run: prompt_scoring.score(
                    result, error, product, name, run, spec.categories
                ),
                sink=progress,
            )
    except RuntimeError as exc:
        console.print(escape(str(exc)))
        raise SystemExit(1) from exc

    if case == "agent":
        summary = benchmark.write_outputs(rows, directory, stamp, model, settings, runs)
        render_bench(benchmark.summarize(rows))
    else:
        summary = directory / f"summary-prompt-{stamp}.md"
        summary.write_text(
            prompt_scoring.summary_markdown(rows, spec.categories, model, settings, runs, stamp)
        )
        render_table(
            "Structured prompt bench",
            prompt_scoring.SUMMARY_COLUMNS,
            [prompt_scoring.summary_cells(line) for line in prompt_scoring.summarize(rows)],
        )
    console.print(f"{sink.path}\n{summary}")


@app.command
def tools() -> None:
    """List the tools every variant shares."""
    for tool in TOOLS:
        console.print(f"[bold]{tool.__name__}[/bold] — {(tool.__doc__ or '').splitlines()[0]}")
