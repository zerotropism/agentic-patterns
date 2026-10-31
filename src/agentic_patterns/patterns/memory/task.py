"""The conversation memory use case: answer a question about something said turns earlier.

Each conversation states one fact, then goes on with unrelated questions, one of which
states the same kind of fact about someone else, then asks for the fact. The model answers
every turn; what is scored is the last answer. The conversations are generated with a
fixed seed, so every one has an exact reference and a known distractor.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Conversation:
    """The user's turns, in order, and what the last answer must say, and must not."""

    id: str
    slot: str
    distance: int
    turns: tuple[str, ...]
    expected: str
    distractor: str


@dataclass(frozen=True, slots=True)
class Slot:
    """A kind of fact: how it is stated, how it is stated about someone else, how it is asked."""

    values: tuple[str, ...]
    fact: Callable[[str], str]
    other: Callable[[str], str]
    question: str


SLOTS: dict[str, Slot] = {
    "budget": Slot(
        values=("1,200", "1,850", "2,000", "2,750", "3,400", "4,999", "6,500", "8,250"),
        fact=lambda v: f"My budget for a new kettle is ₹{v}.",
        other=lambda v: f"My brother's budget for his kettle was ₹{v}, but that is his business.",
        question="What is my budget for the kettle?",
    ),
    "city": Slot(
        values=("Pune", "Jaipur", "Kochi", "Lucknow", "Indore", "Mysuru", "Nagpur", "Surat"),
        fact=lambda v: f"Please note that I live in {v}.",
        other=lambda v: f"My sister lives in {v}, a long way from me.",
        question="In which city do I live?",
    ),
    "pet": Slot(
        values=("Biscuit", "Mango", "Pepper", "Tofu", "Simba", "Kiwi", "Noodle", "Bruno"),
        fact=lambda v: f"I am buying a bowl for my dog, who is called {v}.",
        other=lambda v: f"My neighbour's dog, {v}, keeps barking at night.",
        question="What is my dog called?",
    ),
    "day": Slot(
        values=("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"),
        fact=lambda v: f"I can only receive deliveries on {v}.",
        other=lambda v: f"My last order came on a {v}, which did not suit me at all.",
        question="On which day can I receive deliveries?",
    ),
}

# Questions the model can answer without the conversation, and that say nothing about it
FILLERS = (
    "What is the capital of Japan?",
    "Give me one tip to keep a steel kettle clean.",
    "How many days are there in a leap year?",
    "What does a Wi-Fi router do?",
    "Suggest a name for a houseplant.",
    "Which is heavier, a kilogram of rice or a kilogram of cotton?",
    "What is 12 multiplied by 12?",
    "Name a fruit that is yellow when ripe.",
    "What do you call a baby cat?",
    "How many minutes are there in two hours?",
)


def build_conversation(
    index: int, slot_name: str, distance: int, rng: random.Random
) -> Conversation:
    """The fact, distance turns of which one is the distractor, then the question.

    The distractor's place among the turns in between is drawn, so it is not always the
    turn just before the question.
    """
    slot = SLOTS[slot_name]
    expected, distractor = rng.sample(slot.values, 2)
    between = rng.sample(FILLERS, distance - 1)
    between.insert(rng.randrange(distance), slot.other(distractor))
    return Conversation(
        id=f"{slot_name}-{distance}-{index}",
        slot=slot_name,
        distance=distance,
        turns=(slot.fact(expected), *between, slot.question),
        expected=expected,
        distractor=distractor,
    )


def build_conversations(distances: list[int], per_distance: int, seed: int) -> list[Conversation]:
    """per_distance conversations at each distance, the slots taken in turn."""
    if any(d < 1 or d > len(FILLERS) + 1 for d in distances):
        raise ValueError(f"Distances go from 1 to {len(FILLERS) + 1}.")
    rng = random.Random(seed)
    names = list(SLOTS)
    return [
        build_conversation(index, names[index % len(names)], distance, rng)
        for distance in distances
        for index in range(per_distance)
    ]


def load_task(config: dict[str, Any]) -> tuple[str, list[Conversation]]:
    """The system prompt and the conversations, from the memory section of config.yaml."""
    section = config["memory"]
    conversations = build_conversations(
        section["distances"], section["per_distance"], section["seed"]
    )
    return section["system"].strip(), conversations
