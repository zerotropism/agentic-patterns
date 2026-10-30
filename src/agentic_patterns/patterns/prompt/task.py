"""The structured prompt use case: put a product in one of a few shop categories.

Everything the three variants share lives here: the products, the template text and the
arguments that fill it. How the template becomes messages is what each variant shows.
"""

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_patterns.config import ROOT


@dataclass(frozen=True, slots=True)
class Product:
    """One row of the shop file, reduced to what the prompt shows and what it is scored on."""

    id: str
    name: str
    description: str
    category: str


@dataclass(frozen=True, slots=True)
class PromptSpec:
    """The template, as text with {placeholders}, and the examples it shows the model.

    MCP prompts carry no system role, only user and assistant messages. So that the three
    variants can send the same request, the instructions are the first user message.
    """

    instructions: str
    user: str
    categories: tuple[str, ...]
    examples: tuple[Product, ...]

    def instruction_arguments(self) -> dict[str, str]:
        return {"categories": ", ".join(self.categories)}


def top_category(path: str) -> str:
    """Computers&Accessories|Accessories&Peripherals|...|USBCables gives Computers&Accessories."""
    return path.split("|", 1)[0]


def read_products(path: Path, description_chars: int) -> list[Product]:
    """Every distinct product of the file, in file order, description cut to a fixed length.

    The file repeats a product once per review: the first occurrence is kept.
    """
    seen: dict[str, Product] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["product_id"] in seen:
                continue
            seen[row["product_id"]] = Product(
                id=row["product_id"],
                name=row["product_name"].strip(),
                description=" ".join(row["about_product"].split())[:description_chars],
                category=top_category(row["category"]),
            )
    return list(seen.values())


def select(
    products: list[Product], categories: tuple[str, ...], per_category: int, seed: int
) -> tuple[tuple[Product, ...], list[Product]]:
    """The examples, then the sample to score.

    The examples are the first product of each category, in file order; the sample draws
    per_category others from each category with a fixed seed. Both are reproducible and
    never overlap, so the model is never scored on a product it was shown.
    """
    by_category: dict[str, list[Product]] = defaultdict(list)
    for product in products:
        if product.category in categories:
            by_category[product.category].append(product)

    missing = [c for c in categories if not by_category[c]]
    if missing:
        raise ValueError(f"No product in the file for: {', '.join(missing)}.")

    examples = tuple(by_category[c][0] for c in categories)
    rng = random.Random(seed)
    sample = []
    for category in categories:
        pool = by_category[category][1:]
        if len(pool) < per_category:
            raise ValueError(f"{category}: {len(pool)} products, {per_category} asked.")
        sample += rng.sample(pool, per_category)
    return examples, sample


def load_task(config: dict[str, Any]) -> tuple[PromptSpec, list[Product]]:
    """The prompt and the products to classify, from the prompt section of config.yaml."""
    section = config["prompt"]
    categories = tuple(section["categories"])
    products = read_products(ROOT / section["data"], section["description_chars"])
    examples, sample = select(products, categories, section["per_category"], section["seed"])
    spec = PromptSpec(
        instructions=section["instructions"].strip(),
        user=section["user"].strip(),
        categories=categories,
        examples=examples,
    )
    return spec, sample
