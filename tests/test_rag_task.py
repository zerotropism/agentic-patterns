"""The RAG case's catalogue, search and questions."""

import pytest

from agentic_patterns.config import load_config
from agentic_patterns.patterns.rag.task import (
    BM25,
    Document,
    build_questions,
    facts,
    format_documents,
    load_task,
)

DOCS = [
    Document("p0", "Trail Shoes", "Light shoes. Approx. weight: 1 lb. 1 oz. per pair. Imported."),
    Document("p1", "Wool Socks", "Warm socks.\nWeight: 3 oz.\nMade in USA."),
    Document("p2", "Rain Jacket", "Waterproof jacket. Made in the Philippines."),
]


def test_search_ranks_the_matching_document_first() -> None:
    index = BM25(DOCS)
    assert [d.id for d in index.search("how heavy are the wool socks", 2)][0] == "p1"


def test_search_returns_at_most_k_and_nothing_without_a_matching_term() -> None:
    index = BM25(DOCS)
    assert len(index.search("shoes socks jacket", 2)) == 2
    assert index.search("zeppelin", 3) == []


def test_facts_are_read_in_both_weight_forms_and_from_made_in() -> None:
    assert facts(DOCS[0]) == {"weight": "1 lb. 1 oz"}
    assert facts(DOCS[1]) == {"weight": "3 oz", "origin": "USA"}
    assert facts(DOCS[2]) == {"origin": "Philippines"}


def test_the_model_reads_numbered_blocks_name_first() -> None:
    assert format_documents(DOCS[:2]).startswith("[1] Trail Shoes\nLight shoes.")
    assert "\n\n[2] Wool Socks\n" in format_documents(DOCS[:2])


def test_questions_name_the_product_and_skip_duplicate_names() -> None:
    docs = [*DOCS, Document("p3", "Wool Socks", "Weight: 4 oz.")]
    questions = build_questions(docs, 1, seed=1)

    assert {q.kind for q in questions} == {"weight", "origin"}
    assert all(q.document != "p3" and q.document != "p1" for q in questions if q.kind == "weight")
    weight = next(q for q in questions if q.kind == "weight")
    assert weight.text == "How much does the Trail Shoes weigh?"
    assert weight.answer == "1 lb. 1 oz"


def test_too_few_products_for_a_kind_is_refused() -> None:
    with pytest.raises(ValueError, match="weight: 2 products, 5 asked"):
        build_questions(DOCS, 5, seed=1)


def test_the_shipped_configuration_gives_thirty_questions_of_each_kind() -> None:
    task, questions = load_task(load_config())

    assert len(task.index.documents) == 1000
    assert len(questions) == len({q.id for q in questions}) == 60
    assert {sum(q.kind == k for q in questions) for k in ("weight", "origin")} == {30}
    assert build_questions(task.index.documents, 30, seed=3) == questions
