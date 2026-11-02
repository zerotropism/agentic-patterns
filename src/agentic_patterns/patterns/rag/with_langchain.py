"""Same pipeline, with LangChain 1.x: a BaseRetriever over the same BM25, in an LCEL chain.

retriever | format, the question passed through, the prompt template, the chat model. The
retrieved documents are kept in the chain's output, so the bench can see what the model saw.
"""

import time
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document as LcDocument
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_ollama import ChatOllama

from agentic_patterns.models import DEFAULT_SETTINGS, GenerationSettings, RunResult
from agentic_patterns.patterns.rag.raw import search_call
from agentic_patterns.patterns.rag.task import BM25, Document, Question, RagTask, format_documents


class Bm25Retriever(BaseRetriever):
    """The shared BM25, behind LangChain's retriever interface."""

    index: Any
    k: int = 3

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[LcDocument]:
        return [
            LcDocument(page_content=d.description, metadata={"id": d.id, "name": d.name})
            for d in self.index.search(query, self.k)
        ]


def as_documents(documents: list[LcDocument]) -> list[Document]:
    return [Document(d.metadata["id"], d.metadata["name"], d.page_content) for d in documents]


def build_chain(task: RagTask, chat_model: Any, index: BM25 | None = None):
    retriever = Bm25Retriever(index=index or task.index, k=task.k)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", task.instructions),
            ("human", "Product descriptions:\n\n{context}\n\nQuestion: {question}"),
        ]
    )
    context = RunnableLambda(lambda inputs: format_documents(as_documents(inputs["documents"])))
    return RunnableParallel(documents=retriever, question=RunnablePassthrough()).assign(
        answer=RunnablePassthrough.assign(context=context) | prompt | chat_model
    )


class LangChainRag:
    name = "langchain"

    def __init__(
        self,
        model: str,
        task: RagTask,
        settings: GenerationSettings = DEFAULT_SETTINGS,
        chat_model: Any = None,
    ) -> None:
        self.model = model
        self.task = task
        self.settings = settings
        # chat_model is injected by the tests; reasoning is ChatOllama's name for think
        self.chat_model = chat_model or ChatOllama(
            model=model, temperature=settings.temperature, reasoning=settings.think
        )
        self.chain = build_chain(task, self.chat_model)

    async def run(self, question: Question) -> RunResult:
        started = time.perf_counter()
        output = await self.chain.ainvoke(question.text)
        message = output["answer"]
        usage = getattr(message, "usage_metadata", None) or {}
        ids = [d.metadata["id"] for d in output["documents"]]
        return RunResult(
            variant=self.name,
            question=question.id,
            answer=str(message.content),
            tool_calls=[search_call(question.text, self.task.k, ids)],
            model_calls=1,
            duration_seconds=time.perf_counter() - started,
            prompt_tokens=usage.get("input_tokens"),
        )
