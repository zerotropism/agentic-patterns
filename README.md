# agentic-patterns

The same agentic use case, implemented three ways, side by side: a hand-written tool loop, the
same thing built with LangChain 1.x, and the same thing again with the tools served over MCP.

The point is not which one wins. It is what you have to write, what the framework writes for
you, and whether that changes the outcome.

## Try it

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/) and a running [Ollama](https://ollama.com/).

```bash
uv sync
ollama pull llama3.2:3b

uv run agentic-patterns tools                     # what the three variants share
uv run agentic-patterns run combined --variant raw
uv run agentic-patterns compare --runs 5          # every variant, every question
```

The model comes from `config.yaml` and can be overridden per run:

```bash
AGENTIC_PATTERNS_MODEL=qwen3:8b uv run agentic-patterns compare
```

## The three variants

All three answer the same questions, with the same tools, the same system prompt and the same
model, and return the same `RunResult`. Only the machinery differs.

| | tools | schemas | loop |
|---|---|---|---|
| `raw` | declared in a dict | written by hand from the signature | written by hand |
| `langchain` | passed as plain callables | derived by the framework | provided by the framework |
| `mcp` | served by a FastMCP server | served with the tools | written by hand |

`raw` is the baseline: roughly forty lines of schema generation, dispatch, iteration cap and
error feedback. `langchain` replaces all of it with one `create_agent` call, and adds the work
of digging the tool calls back out of a message list. `mcp` sits in between — the schema comes
with the tool, the loop does not — and that loop is the one you would write to drive any
third-party MCP server.

## What the tools are

Two plain Python functions, shared by every variant: `today()` and `calculate(expression)`.
`calculate` evaluates arithmetic through an AST walk that accepts numbers and operators and
nothing else — no names, no calls, no attributes. It replaces both the `llm-math` chain, which
had the model do the arithmetic, and `PythonREPLTool`, which executed whatever the model wrote.

Their docstrings are the descriptions sent to the model, in all three variants. One place
describes each tool.

## What running it actually shows

At the time of writing, on `llama3.2:3b`, the three variants are indistinguishable in capability:
each one answers all three questions, each takes two model round trips, each lands within a few
hundred milliseconds of the others.

What is clearly visible is that **the same variant does not behave the same way twice**.
`temperature: 0.0` does not make Ollama deterministic. Across small samples the same question
produced two tool calls, one, or none, depending on the run — and "none" is the failure mode
that matters: the model answered the arithmetic question itself instead of calling the
calculator, which is exactly what the system prompt forbids and exactly the case where its
answer cannot be trusted.

So the honest reading is: with this model and these tools, the framework does not buy capability.
It buys not writing the loop, and not getting the schema subtly wrong.

Comparing the variants quantitatively needs many runs and a confidence interval, not the three
or five this repo currently does. That is what `bench.py` is for, and it is not written yet.

## Structure

```
src/agentic_patterns/
├── tools.py          the shared tool set, as plain functions
├── models.py         ToolCall and RunResult — what every variant returns
├── protocols.py      Pattern
├── patterns/agent/
│   ├── raw.py             ollama client, hand-written loop
│   ├── with_langchain.py  create_agent
│   └── with_mcp.py        FastMCP server, driven through a Client
├── rendering.py      tables
└── cli.py            cyclopts commands
```

Adding a variant means adding a class with a `name` and an `async run`, plus one line in
`patterns/agent/__init__.py`. Adding a use case means a sibling package under `patterns/`.

An agent holds a client bound to the running event loop, so it must be built and used inside a
single `asyncio.run`. The CLI does that; a script that calls `asyncio.run` once per question
will fail on the second one.

## Tests

```bash
uv run pytest
```

No Ollama: the LangChain variant runs against a fake chat model, the MCP variant against an
in-memory server, and the raw variant's loop is exercised through its tool dispatch. One test
asserts that the raw and MCP schemas are identical — if they diverge, the comparison measures
the schema instead of the approach.

## History

This repo was a tour of LangChain 0.3 written in an object-oriented style. LangChain 1.x removed
most of what it was built on (`initialize_agent`, `AgentType`, `load_tools`, `LLMChain`, the
legacy memories), which made it a choice between redoing the tour on the new API and asking a
question that does not expire with the next major version. The old modules are in the git
history.
