"""The conversation memory case's conversations: structure, distractor, reproducibility."""

import random

import pytest

from agentic_patterns.config import load_config
from agentic_patterns.patterns.memory.task import (
    FILLERS,
    SLOTS,
    build_conversation,
    build_conversations,
    load_task,
)


@pytest.mark.parametrize("slot", sorted(SLOTS))
@pytest.mark.parametrize("distance", [1, 3, 6])
def test_fact_first_question_last_and_the_distractor_in_between(slot, distance) -> None:
    conversation = build_conversation(0, slot, distance, random.Random(1))
    turns = conversation.turns

    assert len(turns) == distance + 2
    assert turns[0] == SLOTS[slot].fact(conversation.expected)
    assert turns[-1] == SLOTS[slot].question
    assert turns[1:-1].count(SLOTS[slot].other(conversation.distractor)) == 1
    assert conversation.expected != conversation.distractor


def test_the_turns_in_between_say_nothing_about_the_fact() -> None:
    conversation = build_conversation(0, "city", 6, random.Random(2))
    between = [t for t in conversation.turns[1:-1] if t in FILLERS]
    assert len(between) == 5
    assert not any(conversation.expected in turn for turn in between)


def test_the_conversations_are_the_same_for_the_same_seed() -> None:
    first = build_conversations([1, 3], 4, seed=5)
    assert build_conversations([1, 3], 4, seed=5) == first
    assert build_conversations([1, 3], 4, seed=6) != first


def test_slots_are_taken_in_turn_at_each_distance() -> None:
    conversations = build_conversations([1, 3], 4, seed=5)
    assert [c.slot for c in conversations] == list(SLOTS) * 2
    assert [c.distance for c in conversations] == [1] * 4 + [3] * 4


def test_a_distance_beyond_the_fillers_is_refused() -> None:
    with pytest.raises(ValueError, match="Distances go from 1 to"):
        build_conversations([len(FILLERS) + 2], 1, seed=1)


def test_the_shipped_configuration_gives_twelve_conversations_per_distance() -> None:
    system, conversations = load_task(load_config())

    assert "short sentences" in system
    assert len(conversations) == len({c.id for c in conversations}) == 36
    assert {sum(c.distance == d for c in conversations) for d in (1, 3, 6)} == {12}
