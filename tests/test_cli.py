"""CLI plumbing: variant resolution and question lookup. No model involved."""

import pytest

from agentic_patterns.cli import app, build_agent, questions_from, select_questions
from agentic_patterns.config import load_config
from agentic_patterns.models import GenerationSettings
from agentic_patterns.patterns.agent import VARIANTS
from agentic_patterns.protocols import Pattern

CONFIG = {
    "model": {"name": "fake", "temperature": 0.3, "think": True},
    "agent": {
        "system_prompt": "p",
        "questions": {
            "date": {"text": "What is today's date?", "tools": ["today"], "expect": ["{today}"]},
            "math": {"text": "What is 2 + 2?"},
        },
    },
}


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_every_variant_can_be_built(variant: str) -> None:
    assert build_agent(variant, CONFIG).name == variant


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_every_variant_is_a_pattern(variant: str) -> None:
    assert isinstance(build_agent(variant, CONFIG), Pattern)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_every_variant_gets_the_configured_settings(variant: str) -> None:
    """The bias this guards against: raw and mcp used to send no temperature at all."""
    assert build_agent(variant, CONFIG).settings == GenerationSettings(temperature=0.3, think=True)


def test_unknown_variant_lists_the_available_ones() -> None:
    with pytest.raises(ValueError, match="raw"):
        build_agent("nope", CONFIG)


def test_a_named_question_is_looked_up() -> None:
    assert questions_from(CONFIG, "date") == ["What is today's date?"]


def test_an_unknown_name_is_taken_as_the_question_itself() -> None:
    assert questions_from(CONFIG, "how tall is Everest?") == ["how tall is Everest?"]


def test_no_name_runs_every_configured_question() -> None:
    assert len(questions_from(CONFIG, None)) == 2


def test_checks_come_with_the_question() -> None:
    date_question = select_questions(CONFIG, "date")[0]
    assert (date_question.tools, date_question.expect) == (("today",), ("{today}",))


def test_a_question_without_checks_has_none() -> None:
    math = select_questions(CONFIG, "math")[0]
    assert (math.tools, math.expect) == (None, ())


def test_the_shipped_configuration_checks_every_question() -> None:
    """A question without expected values would score zero on every run."""
    questions = select_questions(load_config(), None)
    assert questions and all(question.checked for question in questions)


def test_a_literal_question_is_not_checked() -> None:
    assert not select_questions(CONFIG, "how tall is Everest?")[0].checked


def test_a_question_without_tools_key_is_not_checked_but_an_empty_list_is() -> None:
    config = {"agent": {"questions": {"a": {"text": "a", "expect": ["x"]}}}}
    assert select_questions(config, "a")[0].tools is None
    config["agent"]["questions"]["a"]["tools"] = []
    assert select_questions(config, "a")[0].checked


def test_bench_refuses_a_question_it_cannot_score(tmp_path) -> None:
    with pytest.raises(SystemExit) as exit_info:
        app(["bench", "--question", "how tall is Everest?", "--output", str(tmp_path)])
    assert exit_info.value.code == 1
    assert not list(tmp_path.iterdir())


def test_bench_refuses_an_unknown_case(tmp_path) -> None:
    with pytest.raises(SystemExit) as exit_info:
        app(["bench", "--case", "nope", "--output", str(tmp_path)])
    assert exit_info.value.code == 1


def test_a_question_applies_to_the_agent_case_only(tmp_path) -> None:
    with pytest.raises(SystemExit) as exit_info:
        app(["bench", "--case", "prompt", "--question", "date", "--output", str(tmp_path)])
    assert exit_info.value.code == 1
    assert not list(tmp_path.iterdir())


def test_classify_refuses_a_product_outside_the_sample() -> None:
    with pytest.raises(SystemExit) as exit_info:
        app(["classify", "NOT-A-PRODUCT"])
    assert exit_info.value.code == 1
