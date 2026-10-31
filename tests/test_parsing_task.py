"""The typed parsing case's data: figures, phrasings, sample."""

import pytest

from agentic_patterns.config import load_config
from agentic_patterns.patterns.parsing.task import (
    PHRASINGS,
    Source,
    amount,
    build_items,
    load_task,
    number,
    plain,
    read_sources,
    shorten,
)

KETTLE = Source("K", "Kettle", "₹1,099", "₹399", "64%", "4.2", "24,269")
COLUMNS = ("product_id", "product_name", "actual_price", "discounted_price")
HEADER = ",".join([*COLUMNS, "discount_percentage", "rating", "rating_count"]) + "\n"


def test_figures_are_read_as_the_file_writes_them() -> None:
    assert (number("₹1,099"), number("64%"), number("24,269")) == (1099.0, 64.0, 24269.0)
    assert (plain("₹1,099"), plain("₹1,099.50"), amount("₹1,099")) == ("1099", "1099.5", "1,099")


def test_the_reference_is_typed_from_the_columns() -> None:
    truth = KETTLE.truth
    assert (truth.actual_price, truth.discounted_price) == (1099.0, 399.0)
    assert (truth.discount_percent, truth.rating, truth.rating_count) == (64, 4.2, 24269)


@pytest.mark.parametrize("phrasing", sorted(PHRASINGS))
def test_every_phrasing_writes_every_figure(phrasing: str) -> None:
    text = PHRASINGS[phrasing](KETTLE)
    assert "Kettle" in text and "4.2" in text
    assert ("1,099" in text or "1099" in text) and "399" in text
    assert "64" in text and ("24,269" in text or "24269" in text)


def test_a_long_name_is_cut_at_a_space() -> None:
    assert shorten("Electric   Kettle 1500W Auto Cut-Off", 20) == "Electric Kettle"


def test_duplicates_and_unreadable_figures_are_left_out(tmp_path) -> None:
    path = tmp_path / "shop.csv"
    path.write_text(
        HEADER
        + 'A,Cable,"₹1,099",₹399,64%,4.2,"24,269"\n'
        + 'A,Cable,"₹1,099",₹399,64%,4.2,"24,269"\n'
        + "B,Lamp,₹500,₹250,50%,|,10\n"
        + "C,Mug,₹300,₹150,50%,4.0,\n",
        encoding="utf-8",
    )
    assert [s.id for s in read_sources(path, 100)] == ["A"]


def test_phrasings_are_taken_in_turn_and_the_sample_is_reproducible() -> None:
    sources = [Source(str(i), f"p{i}", "₹200", "₹100", "50%", "4.0", "10") for i in range(30)]
    items = build_items(sources, 8, seed=3)

    assert [i.phrasing for i in items] == list(PHRASINGS) * 2
    assert [i.id for i in build_items(sources, 8, seed=3)] == [i.id for i in items]


def test_too_few_products_is_refused() -> None:
    with pytest.raises(ValueError, match="1 readable products, 2 asked"):
        build_items([KETTLE], 2, seed=1)


def test_the_shipped_configuration_gives_25_listings_per_phrasing() -> None:
    instructions, items = load_task(load_config())

    assert "number of ratings" in instructions
    assert len(items) == len({i.id for i in items}) == 100
    assert {sum(i.phrasing == p for i in items) for p in PHRASINGS} == {25}
