"""Reloading the model between bench phases, with a fake Ollama client."""

from types import SimpleNamespace

import pytest

from agentic_patterns.reload import loaded, reload_model


class FakeClient:
    def __init__(self, listed_for: int) -> None:
        self.calls: list[tuple] = []
        self.listed_for = listed_for

    async def generate(self, **kwargs) -> None:
        self.calls.append(("generate", kwargs))

    async def ps(self) -> SimpleNamespace:
        self.calls.append(("ps",))
        self.listed_for -= 1
        models = [SimpleNamespace(model="llama3.2:3b", name="llama3.2:3b")]
        return SimpleNamespace(models=models if self.listed_for >= 0 else [])


async def test_the_model_is_unloaded_then_loaded_once_gone() -> None:
    client = FakeClient(listed_for=2)
    await reload_model("llama3.2:3b", client, poll=0)
    assert client.calls[0] == ("generate", {"model": "llama3.2:3b", "keep_alive": 0})
    assert [c[0] for c in client.calls[1:-1]] == ["ps"] * 3
    assert client.calls[-1] == ("generate", {"model": "llama3.2:3b"})


async def test_a_model_that_never_unloads_times_out() -> None:
    with pytest.raises(TimeoutError):
        await reload_model("llama3.2:3b", FakeClient(listed_for=10**6), limit=0.05, poll=0)


def test_an_untagged_name_matches_latest() -> None:
    assert loaded("llama3.2", "llama3.2:latest") and not loaded("llama3.2", "llama3.2:3b")
