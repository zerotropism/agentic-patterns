"""The judge task: LLMBar's pairs, each in both orders, and the rubric."""

from agentic_patterns.config import load_config
from agentic_patterns.patterns.judge.task import A, B, judgements, load_task, render

PAIR = {
    "id": "natural-000",
    "set": "natural",
    "subset": "natural",
    "input": "Say hi.",
    "output_1": "Hi.",
    "output_2": "Goodbye.",
    "label": 1,
}


def test_each_pair_is_judged_in_both_orders() -> None:
    first, swapped = judgements(PAIR)
    assert (first.id, swapped.id) == ("natural-000:ab", "natural-000:ba")
    assert (first.output_a, first.output_b, first.expected) == ("Hi.", "Goodbye.", A)
    assert (swapped.output_a, swapped.output_b, swapped.expected) == ("Goodbye.", "Hi.", B)


def test_label_two_means_the_second_output() -> None:
    first, swapped = judgements(PAIR | {"label": 2})
    assert (first.expected, swapped.expected) == (B, A)


def test_the_sample_is_natural_then_25_of_each_adversarial_set() -> None:
    _, items = load_task(load_config())
    assert len(items) == 400
    subsets = [j.subset for j in items if j.order == "ab"]
    assert subsets.count("natural") == 100
    assert all(subsets.count(s) == 25 for s in ("neighbor", "gptinst", "gptout", "manual"))
    # Swapping balances the labels: a judge that always says (a) scores half
    assert sum(j.expected == A for j in items) == 200


def test_the_rubric_shows_both_outputs_and_how_to_answer() -> None:
    prompts, items = load_task(load_config())
    text = render(prompts["json"], items[0])
    assert items[0].output_a in text and items[0].output_b in text
    assert text.endswith('Answer with JSON: {"better": "Output (a)"} or {"better": "Output (b)"}.')
    assert render(prompts["tool"], items[0]).endswith('"Output (a)" or "Output (b)".')
    assert "Do NOT provide any explanation" in text
