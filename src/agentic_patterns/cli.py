"""Command line entry point. One command per thing you might want to compare."""

from typing import Annotated

from cyclopts import App, Parameter

from agentic_patterns.config import load_config, model_name
from agentic_patterns.models import RunResult
from agentic_patterns.patterns.agent import VARIANTS
from agentic_patterns.rendering import console, render_comparison, render_result
from agentic_patterns.tools import TOOLS

app = App(
    name="agentic-patterns",
    help="The same agentic use cases, implemented three ways.",
    version="0.1.0",
)

Variant = Annotated[str, Parameter(help=f"One of: {', '.join(VARIANTS)}.")]
ConfigPath = Annotated[str | None, Parameter(help="Path to the YAML configuration.")]


def build_agent(variant: str, config: dict):
    """Build one variant. Agents are reused across questions: they hold a client."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Use one of: {sorted(VARIANTS)}.")
    return VARIANTS[variant](model_name(config), config["agent"]["system_prompt"])


def questions_from(config: dict, name: str | None) -> list[str]:
    """Named question from the configuration, all of them, or a literal question."""
    configured = config["agent"]["questions"]
    if name is None:
        return list(configured.values())
    return [configured.get(name, name)]


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


@app.command
async def compare(
    *,
    question: str | None = None,
    runs: Runs = 3,
    config: ConfigPath = None,
) -> None:
    """Run every variant on the configured questions and tabulate what they did.

    A single run proves nothing: the same model, at temperature 0, does not always call
    the same tools. Repeat and read the rates.
    """
    loaded = load_config(config)
    texts = questions_from(loaded, question)
    results: list[RunResult] = []

    for name in VARIANTS:
        agent = build_agent(name, loaded)
        for text in texts:
            for _ in range(runs):
                results.append(await agent.run(text))

    render_comparison(results)


@app.command
def tools() -> None:
    """List the tools every variant shares."""
    for tool in TOOLS:
        console.print(f"[bold]{tool.__name__}[/bold] — {(tool.__doc__ or '').splitlines()[0]}")
