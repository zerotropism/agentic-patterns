"""The typed parsing use case: read the figures of a product listing into a typed object.

The listings are written from the columns of the shop file, in four phrasings, so every
field has an exact reference. What the three variants share lives here: the target type,
the instructions, the listings.
"""

import csv
import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agentic_patterns.config import ROOT


class Listing(BaseModel):
    """What the model must produce. The descriptions are sent to the model with the schema."""

    actual_price: float = Field(description="Price before discount, in rupees, as a number")
    discounted_price: float = Field(description="Price after discount, in rupees, as a number")
    discount_percent: int = Field(ge=0, le=100, description="Discount, in percent")
    rating: float = Field(ge=0, le=5, description="Average customer rating, out of 5")
    rating_count: int = Field(ge=0, description="Number of customer ratings")


FIELDS = tuple(Listing.model_fields)


def with_schema(instructions: str) -> str:
    """The instructions followed by the JSON schema, descriptions included.

    Ollama's format= constrains the answer to the schema without showing it to the model,
    which sees the field names only, as it is made to write them. Measured on 31 October
    2026: qwen3.5:4b-mlx read actual_price as the price actually paid, and got it right 0
    times in 20; with the schema written here, 20 times. A tool's schema is in the prompt
    already, so the MCP variant does not need this.
    """
    schema = json.dumps(Listing.model_json_schema())
    return f"{instructions}\n\nAnswer with JSON that matches this schema:\n{schema}"


@dataclass(frozen=True, slots=True)
class Item:
    """One listing to parse, the phrasing it was written in, and the exact answer."""

    id: str
    phrasing: str
    text: str
    truth: Listing


@dataclass(frozen=True, slots=True)
class Source:
    """The columns of one product, as the file writes them (₹1,099, 64%, 24,269)."""

    id: str
    name: str
    actual: str
    discounted: str
    discount: str
    rating: str
    count: str

    @property
    def truth(self) -> Listing:
        return Listing(
            actual_price=number(self.actual),
            discounted_price=number(self.discounted),
            discount_percent=int(number(self.discount)),
            rating=number(self.rating),
            rating_count=int(number(self.count)),
        )


def number(text: str) -> float:
    return float(text.replace("₹", "").replace(",", "").replace("%", "").strip())


def plain(text: str) -> str:
    """₹1,099 becomes 1099: the same figure without currency sign or thousands separator."""
    value = number(text)
    return f"{value:.2f}".rstrip("0").rstrip(".") if value % 1 else str(int(value))


def amount(text: str) -> str:
    """₹1,099 becomes 1,099."""
    return text.replace("₹", "")


# The order of the figures changes from one phrasing to the next, and so does their format
PHRASINGS: dict[str, Callable[[Source], str]] = {
    "was_now": lambda s: (
        f"{s.name}. Was ₹{amount(s.actual)}, now ₹{amount(s.discounted)} ({s.discount} off). "
        f"Rated {s.rating} out of 5 by {s.count} customers."
    ),
    "deal": lambda s: (
        f"Deal of the day: {s.name} for just ₹{amount(s.discounted)} instead of "
        f"₹{amount(s.actual)}, that is {s.discount} less. {s.count} ratings, {s.rating}/5 "
        "on average."
    ),
    "plain": lambda s: (
        f"{s.name} | List price: Rs {plain(s.actual)} | Offer price: Rs {plain(s.discounted)}"
        f" | You save {plain(s.discount)} percent | {s.rating} stars from {plain(s.count)} reviews"
    ),
    "sheet": lambda s: (
        f"Product: {s.name}\nMRP: INR {amount(s.actual)}\nSelling price: INR "
        f"{amount(s.discounted)}\nDiscount: {s.discount}\nRating: {s.rating} ({s.count} ratings)"
    ),
}


def shorten(name: str, chars: int) -> str:
    """At most chars characters, cut at a space so that no word is split."""
    name = " ".join(name.split())
    return name if len(name) <= chars else name[:chars].rsplit(" ", 1)[0]


def read_sources(path: Path, name_chars: int) -> list[Source]:
    """Every distinct product whose five figures are readable, in file order."""
    seen: dict[str, Source] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["product_id"] in seen:
                continue
            source = Source(
                id=row["product_id"],
                name=shorten(row["product_name"], name_chars),
                actual=row["actual_price"],
                discounted=row["discounted_price"],
                discount=row["discount_percentage"],
                rating=row["rating"],
                count=row["rating_count"],
            )
            try:
                source.truth  # noqa: B018 - a figure the file got wrong leaves the product out
            except ValueError:
                continue
            seen[source.id] = source
    return list(seen.values())


def build_items(sources: list[Source], count: int, seed: int) -> list[Item]:
    """count products drawn with a fixed seed, the phrasings taken in turn."""
    if len(sources) < count:
        raise ValueError(f"{len(sources)} readable products, {count} asked.")
    chosen = random.Random(seed).sample(sources, count)
    names = list(PHRASINGS)
    return [
        Item(
            id=source.id,
            phrasing=names[index % len(names)],
            text=PHRASINGS[names[index % len(names)]](source),
            truth=source.truth,
        )
        for index, source in enumerate(chosen)
    ]


def load_task(config: dict[str, Any]) -> tuple[str, list[Item]]:
    """The instructions and the listings, from the parsing section of config.yaml."""
    section = config["parsing"]
    sources = read_sources(ROOT / section["data"], section["name_chars"])
    return section["instructions"].strip(), build_items(
        sources, section["products"], section["seed"]
    )
