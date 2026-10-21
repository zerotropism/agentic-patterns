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
uv run agentic-patterns bench                     # every variant, every question, 20 runs
```

The model comes from `config.yaml` and can be overridden per run:

```bash
AGENTIC_PATTERNS_MODEL=qwen3.5:4b-mlx uv run agentic-patterns bench
```

## The three variants

All three answer the same questions, with the same tools, the same system prompt, the same
model and the same sampling settings (`temperature` and `think`, from `config.yaml`), and return
the same `RunResult`. Only the machinery differs. A test checks that `raw` and `mcp` send the model
byte-identical requests, and another that `langchain` passes the same settings to `ChatOllama`.

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

## Bench

`agentic-patterns bench` runs every variant on every configured question, 20 times by default,
and scores each run on two counts, independently:

- tools: every tool the question needs was called and returned without error;
- answer: every expected value is in the final answer. The date counts in its usual spellings
  (`2026-10-21`, `October 21, 2026`, `21 October 2026`...), numbers with or without thousands
  separators.

A run that answers right without calling the tool is the case the system prompt forbids, which
is why the two counts are kept apart. Variants are interleaved, their order rotates from one run
to the next, and one warm-up run per variant is discarded. A run that raises or exceeds the
timeout counts as failed. Each outcome is written as it happens to `evaluation/outcomes-*.jsonl`;
the summary, rates with their 95% Wilson interval, goes to `evaluation/summary-*.md`.

What a question checks lives next to it in `config.yaml`:

```yaml
combined:
  text: "What is today's date, and what is 365 * 3?"
  tools: [today, calculate]
  expect: ["{today}", "1095"]
```

### Results

Measured on 21 October 2026, on a Mac with Ollama, temperature 0 and `think: false`, 20 runs per
question and variant, 180 runs per model:

| model | variant | tools ok | answer ok | errors | model calls | s median | s p90 |
|---|---|---|---|---|---|---|---|
| `llama3.2:3b` | raw | 60/60 | 60/60 | 0 | 2.0 | 1.4 | 1.9 |
| `llama3.2:3b` | langchain | 60/60 | 60/60 | 0 | 2.0 | 1.4 | 1.9 |
| `llama3.2:3b` | mcp | 60/60 | 60/60 | 0 | 2.0 | 1.4 | 1.9 |
| `qwen3.5:4b-mlx` | raw | 60/60 | 60/60 | 0 | 2.0 | 2.9 | 3.3 |
| `qwen3.5:4b-mlx` | langchain | 60/60 | 60/60 | 0 | 2.0 | 2.9 | 3.2 |
| `qwen3.5:4b-mlx` | mcp | 60/60 | 60/60 | 0 | 2.0 | 2.9 | 3.3 |

60/60 has a 95% interval of 0.94 to 1.00. Every question taken alone scored 20/20 for every
variant and both models; the per-question lines are in `evaluation/`.

What it shows:

- The three variants are indistinguishable. Same success, same two model calls per run, latency
  within 0.1 s. Neither the in-process MCP round trip nor the LangChain graph costs anything
  visible next to the model's own time.
- No run of the 360 skipped a tool or answered wrong. An earlier version of this README reported
  the opposite: the same question produced two tool calls, one or none from one run to the next.
  Those observations were made while `raw` and `mcp` sent no temperature, so Ollama sampled with
  the model's default rather than 0, and `langchain` was the only variant at 0. With identical
  settings, that variability is gone.
- The bench is saturated. Three questions that every variant always gets right cannot rank the
  variants; harder questions are needed for that, such as several tool calls in sequence, a tool
  error to recover from, or a question no tool answers.

So, on these questions, the framework does not buy capability. It buys not writing the loop and
not getting the schema subtly wrong, at no measurable cost in latency.

## Structure

```
src/agentic_patterns/
├── tools.py          the shared tool set, as plain functions
├── models.py         GenerationSettings, Question, ToolCall, RunResult
├── protocols.py      Pattern: a name and an async run, all a variant implements
├── patterns/agent/
│   ├── raw.py             ollama client, hand-written loop
│   ├── with_langchain.py  create_agent
│   └── with_mcp.py        FastMCP server, driven through a Client
├── bench.py          scoring, Wilson intervals, the interleaved run loop
├── rendering.py      tables
└── cli.py            cyclopts commands
evaluation/           bench outcomes and summaries, one pair per measurement
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

No Ollama: the LangChain variant runs against a fake chat model, the raw and MCP loops against a
fake Ollama client, the MCP tools through an in-memory server, and the bench against fake agents.
One test asserts that the raw and MCP variants send the model identical requests, schemas and
settings included. If they diverge, the comparison measures the request instead of the approach.

## History

This repo was a tour of LangChain 0.3 written in an object-oriented style. LangChain 1.x removed
most of what it was built on (`initialize_agent`, `AgentType`, `load_tools`, `LLMChain`, the
legacy memories), which made it a choice between redoing the tour on the new API and asking a
question that does not expire with the next major version. The old modules are in the git
history.
