"""The RAG use case: answer a question about one product of a catalogue of a thousand.

The questions are written from facts the descriptions state in a fixed form (a weight, a
country of manufacture), and name the product, so each one has a reference document and
an exact answer. The search is the same BM25 for the three variants: what differs is who
calls it, and with what query.
"""

import csv
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_patterns.config import ROOT

TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    name: str
    description: str

    @property
    def text(self) -> str:
        return f"{self.name}\n{self.description}"


@dataclass(frozen=True, slots=True)
class Question:
    """A question about one product, its kind, the reference document and the answer."""

    id: str
    kind: str
    text: str
    document: str
    answer: str


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class BM25:
    """Okapi BM25 over the catalogue, k1 1.5 and b 0.75, as in local-rag."""

    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = documents
        self.k1, self.b = k1, b
        self.terms = [Counter(tokens(d.text)) for d in documents]
        self.lengths = [sum(t.values()) for t in self.terms]
        self.average = sum(self.lengths) / len(self.lengths)
        frequency: Counter = Counter(term for t in self.terms for term in t)
        n = len(documents)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in frequency.items()}

    def search(self, query: str, k: int) -> list[Document]:
        """The k best documents, best first; ties keep catalogue order."""
        wanted = [t for t in tokens(query) if t in self.idf]
        scores = []
        for index, terms in enumerate(self.terms):
            norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average)
            score = sum(
                self.idf[t] * terms[t] * (self.k1 + 1) / (terms[t] + norm)
                for t in wanted
                if t in terms
            )
            scores.append((score, index))
        best = sorted(scores, key=lambda pair: (-pair[0], pair[1]))[:k]
        return [self.documents[index] for score, index in best if score > 0]


def read_catalogue(path: Path) -> list[Document]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            Document(id=f"p{row['']}", name=row["name"].strip(), description=row["description"])
            for row in csv.DictReader(handle)
        ]


# "Weight: 2 lb. 5 oz." on a line of its own, or "Approx. weight: 1 lb. 1 oz. per pair"
# inside a sentence, where only the figures and their units belong to the weight
WEIGHT = re.compile(
    r"(?m)(?:^\s*Weight:\s*(.+)$|Approx\. weight:\s*((?:\d+(?:\.\d+)?\s*(?:lbs?|oz|g|kg)\.?\s*)+))"
)
ORIGIN = re.compile(r"Made in (?:the )?([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)?)")


def facts(document: Document) -> dict[str, str]:
    """The facts of one description that a question can be written from."""
    found = {}
    if weight := WEIGHT.search(document.description):
        found["weight"] = (weight.group(1) or weight.group(2)).strip().rstrip(".")
    if origin := ORIGIN.search(document.description):
        found["origin"] = origin.group(1)
    return found


QUESTIONS = {
    "weight": "How much does the {name} weigh?",
    "origin": "Where is the {name} made?",
}


def build_questions(documents: list[Document], per_kind: int, seed: int) -> list[Question]:
    """per_kind questions of each kind, about products whose name is unique in the catalogue."""
    names = Counter(d.name for d in documents)
    pools: dict[str, list[tuple[Document, str]]] = defaultdict(list)
    for document in documents:
        if names[document.name] == 1:
            for kind, value in facts(document).items():
                pools[kind].append((document, value))

    rng = random.Random(seed)
    questions = []
    for kind in QUESTIONS:
        if len(pools[kind]) < per_kind:
            raise ValueError(f"{kind}: {len(pools[kind])} products, {per_kind} asked.")
        for document, value in rng.sample(pools[kind], per_kind):
            questions.append(
                Question(
                    id=f"{kind}-{document.id}",
                    kind=kind,
                    text=QUESTIONS[kind].format(name=document.name),
                    document=document.id,
                    answer=value,
                )
            )
    return questions


def format_documents(documents: list[Document]) -> str:
    """What the model reads: one numbered block per document, name then description."""
    return "\n\n".join(f"[{i}] {d.text}" for i, d in enumerate(documents, start=1))


@dataclass(frozen=True, slots=True)
class RagTask:
    instructions: str
    agent_instructions: str
    k: int
    index: BM25


def load_task(config: dict[str, Any]) -> tuple[RagTask, list[Question]]:
    """The shared settings and index, and the questions, from the rag section of config.yaml."""
    section = config["rag"]
    documents = read_catalogue(ROOT / section["data"])
    task = RagTask(
        instructions=section["instructions"].strip(),
        agent_instructions=section["agent_instructions"].strip(),
        k=section["k"],
        index=BM25(documents),
    )
    return task, build_questions(documents, section["per_kind"], section["seed"])
