"""The structured prompt case's data: reading, deduplication, examples and sample."""

import pytest

from agentic_patterns.config import load_config
from agentic_patterns.patterns.prompt.task import (
    Product,
    load_task,
    read_products,
    select,
    top_category,
)

HEADER = "product_id,product_name,category,about_product\n"


def write_csv(tmp_path, *rows: str):
    path = tmp_path / "shop.csv"
    path.write_text(HEADER + "".join(row + "\n" for row in rows), encoding="utf-8")
    return path


def test_the_reference_is_the_first_segment_of_the_category_path() -> None:
    assert top_category("Computers&Accessories|Cables|USBCables") == "Computers&Accessories"


def test_a_product_reviewed_twice_is_read_once_with_a_cut_description(tmp_path) -> None:
    path = write_csv(
        tmp_path,
        'A1,Cable,Electronics|Cables,"long   description, with spaces"',
        "A1,Cable,Electronics|Cables,second review",
        "B2,Kettle,Home&Kitchen|Kettles,boils",
    )
    products = read_products(path, description_chars=10)

    assert [p.id for p in products] == ["A1", "B2"]
    assert products[0].description == "long descr"
    assert products[0].category == "Electronics"


def make(category: str, count: int) -> list[Product]:
    return [Product(f"{category}{i}", f"n{i}", "d", category) for i in range(count)]


def test_examples_are_the_first_of_each_category_and_never_in_the_sample() -> None:
    products = make("A", 5) + make("B", 5)
    examples, sample = select(products, ("A", "B"), per_category=3, seed=1)

    assert [e.id for e in examples] == ["A0", "B0"]
    assert not {e.id for e in examples} & {p.id for p in sample}
    assert [p.category for p in sample] == ["A"] * 3 + ["B"] * 3


def test_the_sample_is_the_same_for_the_same_seed() -> None:
    products = make("A", 20)
    first = select(products, ("A",), 5, seed=7)[1]
    assert select(products, ("A",), 5, seed=7)[1] == first
    assert select(products, ("A",), 5, seed=8)[1] != first


def test_too_few_products_in_a_category_is_refused() -> None:
    with pytest.raises(ValueError, match="B: 1 products, 3 asked"):
        select(make("A", 5) + make("B", 2), ("A", "B"), 3, seed=1)


def test_a_category_absent_from_the_file_is_refused() -> None:
    with pytest.raises(ValueError, match="No product in the file for: C"):
        select(make("A", 5), ("A", "C"), 1, seed=1)


def test_the_shipped_configuration_gives_a_balanced_sample() -> None:
    spec, sample = load_task(load_config())

    assert len(spec.examples) == len(spec.categories) == 4
    assert len(sample) == len({p.id for p in sample}) == 100
    assert {spec.categories.count(p.category) for p in sample} == {1}
