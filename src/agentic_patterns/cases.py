"""The use cases the bench can run, and how to build each one's variants.

A case knows its items, its variants, how to score a run and how to summarize the runs.
The bench command knows nothing else, so a new case is one more entry in CASES.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from agentic_patterns import bench
from agentic_patterns.config import configured_questions, generation_settings, model_name
from agentic_patterns.models import GenerationSettings, Question, RunResult, ToolCall
from agentic_patterns.patterns.agent import VARIANTS as AGENT_VARIANTS
from agentic_patterns.patterns.memory import VARIANTS as MEMORY_VARIANTS
from agentic_patterns.patterns.memory import scoring as memory_scoring
from agentic_patterns.patterns.memory.task import load_task as load_memory
from agentic_patterns.patterns.parsing import VARIANTS as PARSING_VARIANTS
from agentic_patterns.patterns.parsing import scoring as parsing_scoring
from agentic_patterns.patterns.parsing.task import load_task as load_parsing
from agentic_patterns.patterns.prompt import VARIANTS as PROMPT_VARIANTS
from agentic_patterns.patterns.prompt import scoring as prompt_scoring
from agentic_patterns.patterns.prompt.task import PromptSpec
from agentic_patterns.patterns.prompt.task import load_task as load_prompt
from agentic_patterns.protocols import Pattern

Rows = Sequence[dict[str, Any]]


def restore(row: dict[str, Any]) -> RunResult | None:
    """The RunResult an outcome row was scored from; None for a run that raised."""
    if row.get("error"):
        return None
    return RunResult(
        variant=row["variant"],
        question="",
        answer=row.get("answer", ""),
        tool_calls=[ToolCall(**call) for call in row.get("tool_calls", [])],
        model_calls=row.get("model_calls", 1),
        duration_seconds=row.get("seconds") or 0.0,
        prompt_tokens=row.get("prompt_tokens"),
    )


class CaseError(ValueError):
    """A bench that cannot run as asked: reported in one line, before any model call."""


def pick(registry: dict[str, type], variant: str) -> type:
    if variant not in registry:
        raise CaseError(f"Unknown variant '{variant}'. Use one of: {sorted(registry)}.")
    return registry[variant]


def build_agent(variant: str, config: dict) -> Pattern:
    """Build one variant, with the model and sampling settings every variant shares.

    Agents are reused across questions: they hold a client.
    """
    return pick(AGENT_VARIANTS, variant)(
        model_name(config),
        config["agent"]["system_prompt"],
        settings=generation_settings(config),
    )


def build_classifier(variant: str, config: dict, spec: PromptSpec) -> Pattern:
    """One variant of the structured prompt case, with the shared model and settings."""
    return pick(PROMPT_VARIANTS, variant)(
        model_name(config), spec, settings=generation_settings(config)
    )


def build_parser(variant: str, config: dict, instructions: str) -> Pattern:
    """One variant of the typed parsing case, with the shared model and settings."""
    return pick(PARSING_VARIANTS, variant)(
        model_name(config),
        instructions,
        settings=generation_settings(config),
        schema_in_prompt=config["parsing"].get("schema_in_prompt", True),
    )


def build_memory(variant: str, config: dict, system: str) -> Pattern:
    """One variant of the conversation memory case, with the shared model and settings."""
    return pick(MEMORY_VARIANTS, variant)(
        model_name(config), system, settings=generation_settings(config)
    )


def select_questions(config: dict, name: str | None) -> list[Question]:
    """Named question from the configuration, all of them, or a literal question.

    A literal question has nothing to check against: run answers it, bench refuses it.
    """
    configured = configured_questions(config)
    if name is None:
        return list(configured.values())
    return [configured.get(name, Question(name=name, text=name))]


@dataclass(frozen=True)
class Prepared:
    """A case ready to run: what to run on, with what, and how to score and report it."""

    items: Sequence[Any]
    variants: dict[str, Pattern]
    payload: Callable[[Any], Any]
    score: bench.Score
    label: Callable[[dict[str, Any]], str]
    good: Callable[[dict[str, Any]], bool]
    title: str
    columns: Sequence[str]
    cells: Callable[[Rows], list[list[str]]]
    markdown: Callable[[Rows, str, GenerationSettings, int, str], str]
    # How an outcome row names its item, so that a stored run can be scored again
    row_key: str = ""
    item_key: Callable[[Any], str] = lambda item: ""


@dataclass(frozen=True)
class Case:
    name: str
    # The agent questions are few and each one is worth repeating; the other cases have
    # a hundred items each, and repeating them at temperature 0 adds little
    runs: int
    # Prefix of the outcome and summary files; empty for the agent case, which came first
    prefix: str
    prepare: Callable[[dict, Sequence[str] | None, str | None], Prepared]


def no_question(question: str | None) -> None:
    if question:
        raise CaseError("--question applies to the agent case only.")


def prepare_agent(config: dict, names: Sequence[str] | None, question: str | None) -> Prepared:
    questions = select_questions(config, question)
    unchecked = [q.name for q in questions if not q.checked]
    if unchecked:
        raise CaseError(f"No tools or expect configured for: {', '.join(unchecked)}.")
    variants = {n: build_agent(n, config) for n in names or AGENT_VARIANTS}
    today = date.today()
    return Prepared(
        items=questions,
        variants=variants,
        payload=lambda q: q.text,
        score=lambda result, error, q, name, run: bench.score(result, error, q, name, run, today),
        label=lambda row: row["question"],
        good=lambda row: row["tools_ok"] and row["answer_ok"],
        title="Agent bench",
        columns=bench.SUMMARY_COLUMNS,
        cells=lambda rows: [bench.summary_cells(line) for line in bench.summarize(rows)],
        markdown=lambda rows, model, settings, runs, stamp: bench.summary_markdown(
            bench.summarize(rows), model, settings, runs, stamp
        ),
    )


def prepare_prompt(config: dict, names: Sequence[str] | None, question: str | None) -> Prepared:
    no_question(question)
    spec, products = load_prompt(config)
    variants = {n: build_classifier(n, config, spec) for n in names or PROMPT_VARIANTS}
    return Prepared(
        items=products,
        variants=variants,
        payload=lambda product: product,
        score=lambda result, error, product, name, run: prompt_scoring.score(
            result, error, product, name, run, spec.categories
        ),
        label=lambda row: row["product"],
        good=lambda row: row["format_ok"] and row["category_ok"],
        title="Structured prompt bench",
        columns=prompt_scoring.SUMMARY_COLUMNS,
        cells=lambda rows: [
            prompt_scoring.summary_cells(line) for line in prompt_scoring.summarize(rows)
        ],
        markdown=lambda rows, model, settings, runs, stamp: prompt_scoring.summary_markdown(
            rows, spec.categories, model, settings, runs, stamp
        ),
        row_key="product",
        item_key=lambda product: product.id,
    )


def prepare_parsing(config: dict, names: Sequence[str] | None, question: str | None) -> Prepared:
    no_question(question)
    instructions, items = load_parsing(config)
    variants = {n: build_parser(n, config, instructions) for n in names or PARSING_VARIANTS}
    return Prepared(
        items=items,
        variants=variants,
        payload=lambda item: item,
        score=parsing_scoring.score,
        label=lambda row: f"{row['item']} ({row['phrasing']})",
        good=lambda row: row["all_ok"],
        title="Typed parsing bench",
        columns=parsing_scoring.SUMMARY_COLUMNS,
        cells=lambda rows: [
            parsing_scoring.summary_cells(line) for line in parsing_scoring.summarize(rows)
        ],
        markdown=parsing_scoring.summary_markdown,
        row_key="item",
        item_key=lambda item: item.id,
    )


def prepare_memory(config: dict, names: Sequence[str] | None, question: str | None) -> Prepared:
    no_question(question)
    system, conversations = load_memory(config)
    variants = {n: build_memory(n, config, system) for n in names or MEMORY_VARIANTS}
    return Prepared(
        items=conversations,
        variants=variants,
        payload=lambda conversation: conversation,
        score=memory_scoring.score,
        label=lambda row: row["conversation"],
        good=lambda row: row["right"],
        title="Conversation memory bench",
        columns=memory_scoring.SUMMARY_COLUMNS,
        cells=lambda rows: [
            memory_scoring.summary_cells(line) for line in memory_scoring.summarize(rows)
        ],
        markdown=memory_scoring.summary_markdown,
        row_key="conversation",
        item_key=lambda conversation: conversation.id,
    )


CASES = {
    case.name: case
    for case in (
        Case("agent", runs=20, prefix="", prepare=prepare_agent),
        Case("prompt", runs=1, prefix="prompt-", prepare=prepare_prompt),
        Case("parsing", runs=1, prefix="parsing-", prepare=prepare_parsing),
        Case("memory", runs=1, prefix="memory-", prepare=prepare_memory),
    )
}
