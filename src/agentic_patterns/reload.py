"""Reloading the model between phases of a bench, which empties Ollama's prompt cache."""

import asyncio

import ollama


def loaded(model: str, listed: str | None) -> bool:
    """llama3.2 is listed as llama3.2:latest."""
    if listed is None:
        return False
    return listed in (model, f"{model}:latest")


async def reload_model(
    model: str,
    client: ollama.AsyncClient | None = None,
    limit: float = 60.0,
    poll: float = 0.2,
) -> None:
    """Unload the model, wait until Ollama no longer lists it, load it again.

    Unloading stops the runner, and its prompt cache with it. A request with an empty
    prompt loads the model without generating anything.
    """
    client = client or ollama.AsyncClient()
    await client.generate(model=model, keep_alive=0)
    async with asyncio.timeout(limit):
        while any(loaded(model, m.model or m.name) for m in (await client.ps()).models):
            await asyncio.sleep(poll)
    await client.generate(model=model)
