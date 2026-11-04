"""The LLM-as-judge use case: which of two outputs better follows an instruction?

The pairs come from LLMBar (Zeng et al., 2023, MIT licence, data/LLMBAR-LICENSE), whose
labels were set by the authors so that one output is objectively better: data/llmbar.json
holds its 100 natural pairs and 25 of each of its four adversarial sets, where the worse
output is made to look good. Each pair is judged twice, the outputs swapped the second
time, so that a judge that always picks the first output scores half.

The rubric is LLMBar's Vanilla prompt, so that the rates compare with those it published.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentic_patterns.config import ROOT

A, B = "Output (a)", "Output (b)"
Label = Literal["Output (a)", "Output (b)"]
ORDERS = ("ab", "ba")


class Verdict(BaseModel):
    """What the judge must produce: the better output, by its label in the prompt."""

    better: Label = Field(description='The better output: "Output (a)" or "Output (b)"')


@dataclass(frozen=True, slots=True)
class Judgement:
    """One pair in one order: the outputs as shown, and which label is the better one."""

    id: str
    pair: str
    set: str
    subset: str
    order: str
    instruction: str
    output_a: str
    output_b: str
    expected: str


def judgements(pair: dict[str, Any]) -> list[Judgement]:
    """The pair as LLMBar gives it, then swapped. label 1 means output_1 is better."""
    first, second = pair["output_1"], pair["output_2"]
    better_first = pair["label"] == 1
    return [
        Judgement(
            id=f"{pair['id']}:{order}",
            pair=pair["id"],
            set=pair["set"],
            subset=pair["subset"],
            order=order,
            instruction=pair["input"],
            output_a=first if order == "ab" else second,
            output_b=second if order == "ab" else first,
            expected=A if better_first == (order == "ab") else B,
        )
        for order in ORDERS
    ]


def read_pairs(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def render(template: str, judgement: Judgement) -> str:
    return template.format(
        instruction=judgement.instruction,
        output_a=judgement.output_a,
        output_b=judgement.output_b,
    )


def load_task(config: dict[str, Any]) -> tuple[dict[str, str], list[Judgement]]:
    """The prompts and the judgements, from the judge section of config.yaml.

    The prompts are the rubric, followed by how to answer: in JSON for raw and langchain,
    with the verdict tool for mcp.
    """
    section = config["judge"]
    rubric = section["rubric"].strip()
    prompts = {
        "json": f"{rubric}\n\n{section['answer_json'].strip()}",
        "tool": f"{rubric}\n\n{section['answer_tool'].strip()}",
    }
    pairs = read_pairs(ROOT / section["data"])
    return prompts, [j for pair in pairs for j in judgements(pair)]
