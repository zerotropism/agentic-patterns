"""Same conversation, with LangChain 1.x: create_agent and a LangGraph checkpointer.

Each turn sends only the new message; the checkpointer keeps the thread's history and puts
it back in front of the model. This replaces the 0.x memories (ConversationBufferMemory
and its siblings), which LangChain 1.x removed.
"""

import time
import uuid
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

from agentic_patterns.models import DEFAULT_SETTINGS, GenerationSettings, RunResult
from agentic_patterns.patterns.memory.task import Conversation


class LangChainMemory:
    name = "langchain"

    def __init__(
        self,
        model: str,
        system: str,
        settings: GenerationSettings = DEFAULT_SETTINGS,
        chat_model: Any = None,
    ) -> None:
        self.model = model
        self.settings = settings
        # chat_model is injected by the tests; reasoning is ChatOllama's name for think
        self.chat_model = chat_model or ChatOllama(
            model=model, temperature=settings.temperature, reasoning=settings.think
        )
        self.checkpointer = InMemorySaver()
        self.agent = create_agent(
            self.chat_model, [], system_prompt=system, checkpointer=self.checkpointer
        )

    async def run(self, conversation: Conversation) -> RunResult:
        # One thread per run: the checkpointer outlives the call, and a thread reused by a
        # second run would carry the first run's history
        config = {"configurable": {"thread_id": f"{conversation.id}-{uuid.uuid4()}"}}
        started = time.perf_counter()
        state: dict = {"messages": []}

        for turn in conversation.turns:
            state = await self.agent.ainvoke(
                {"messages": [{"role": "user", "content": turn}]}, config
            )

        last = state["messages"][-1]
        usage = getattr(last, "usage_metadata", None) or {}
        return RunResult(
            variant=self.name,
            question=conversation.id,
            answer=str(last.content) if isinstance(last, AIMessage) else "",
            model_calls=len(conversation.turns),
            duration_seconds=time.perf_counter() - started,
            prompt_tokens=usage.get("input_tokens"),
        )
