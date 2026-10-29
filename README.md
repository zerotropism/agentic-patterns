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
Getting `langchain` to that point took two adjustments, measured below: reporting tool errors to
the model, and sending the same tool descriptions.

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

- tools: every tool the question needs was called and returned without error at least once;
  for a question that needs no tool, no tool was called;
- answer: every expected value is in the final answer. The date counts in its usual spellings
  (`2026-10-21`, `October 21, 2026`, `21 October 2026`...), numbers with or without thousands
  separators. A value starting with `=` is computed on the day of the run, with the same
  arithmetic as the `calculate` tool.

Tool calls that returned an error are counted apart, since a run can recover from one. A
question without checks cannot be benched: `bench --question "..."` is refused.

A run that answers right without calling the tool is the case the system prompt forbids, which
is why the two counts are kept apart. Variants are interleaved, their order rotates from one run
to the next, and one warm-up run per variant is discarded. A run that raises or exceeds the
timeout counts as failed. Each outcome is written as it happens to `evaluation/outcomes-*.jsonl`;
the summary, rates with their 95% Wilson interval, goes to `evaluation/summary-*.md`.

What a question checks lives next to it in `config.yaml`:

```yaml
chained:
  text: "How many years are left until the year 2100?"
  tools: [today, calculate]
  expect: ["=2100 - {year}"]
```

| question | what it takes |
|---|---|
| `date` | one call to `today` |
| `math` | one call to `calculate` |
| `combined` | both tools, in the same turn |
| `chained` | `today`, then `calculate` on its result: two turns |
| `recovery` | "What is 2^10 + sqrt(144)?": passed as written, `calculate` fails (`^` is not a power in Python, `sqrt` is not supported), so the error has to reach the model and the model has to rewrite its call |
| `no_tool` | "What is the capital of France?": calling any tool is a failure |

### Results

Measured on 29 October 2026, on a Mac with Ollama, temperature 0 and `think: false`, 20 runs per
question and variant, 360 runs per model (`evaluation/summary-20261029T143717Z.md` and
`summary-20261029T144537Z.md`). The three variants scored the same on every question but one,
so one cell stands for all three; tools ok / answer ok:

| question | `llama3.2:3b` | `qwen3.5:4b-mlx` |
|---|---|---|
| `date` | 20/20 · 20/20 | 20/20 · 20/20 |
| `math` | 20/20 · 20/20 | 20/20 · 20/20 |
| `combined` | 20/20 · 20/20 | 20/20 · 20/20 |
| `chained` | 0/20 · 0/20 | 20/20 · 20/20 |
| `recovery` | 0/20 · 20/20 | 20/20 · 20/20 (raw: tools 19/20) |
| `no_tool` | 0/20 · 20/20 | 20/20 · 20/20 |

Over the six questions:

| model | variant | tools ok | answer ok | errors | tool errors | calls | s median | s p90 |
|---|---|---|---|---|---|---|---|---|
| `llama3.2:3b` | raw | 60/120 | 100/120 | 0 | 40 | 2.0 | 1.2 | 1.9 |
| `llama3.2:3b` | langchain | 60/120 | 100/120 | 0 | 40 | 2.0 | 1.4 | 1.9 |
| `llama3.2:3b` | mcp | 60/120 | 100/120 | 0 | 40 | 2.0 | 1.4 | 1.9 |
| `qwen3.5:4b-mlx` | raw | 119/120 | 120/120 | 0 | 20 | 2.2 | 2.9 | 10.3 |
| `qwen3.5:4b-mlx` | langchain | 120/120 | 120/120 | 0 | 20 | 2.2 | 2.9 | 10.4 |
| `qwen3.5:4b-mlx` | mcp | 120/120 | 120/120 | 0 | 20 | 2.2 | 2.9 | 10.3 |

What the models do, from the outcome files:

- `chained`, `llama3.2:3b`: calls `calculate("2100 - today")`, nesting one tool inside the
  other's argument; after the error, writes a tool call as text instead of making it.
- `recovery`, `llama3.2:3b`: receives the error, then does the arithmetic itself. Right answer,
  forbidden path, which is what the separate tools count is for.
- `no_tool`, `llama3.2:3b`: calls `today` before answering, every time.
- `qwen3.5:4b-mlx` rewrites its failing call (`1024 + 12`) and gets everything right; `recovery`
  costs it three model calls and about 10 s.
- At temperature 0, one run in 360 differed from the 19 others of its question (`qwen`,
  `recovery`, `raw`: the right answer without a successful call).

### What `langchain` needed

Out of the box, `langchain` did not match the other two. Same questions, same models:

| question | model | out of the box, 21 Oct | + tool errors reported, 29 Oct | + same descriptions, 29 Oct |
|---|---|---|---|---|
| `recovery` | `llama3.2:3b` | 20 runs raised | 0/20 · 20/20 | 0/20 · 20/20 |
| `recovery` | `qwen3.5:4b-mlx` | 20 runs raised | 20/20 · 20/20 | 20/20 · 20/20 |
| `chained` | `llama3.2:3b` | 20 runs raised | 0/20 · 0/20 | 0/20 · 0/20 |
| `chained` | `qwen3.5:4b-mlx` | 0/20 · 0/20 | 0/20 · 0/20 | 20/20 · 20/20 |

The last column equals `raw` and `mcp` in every row. The two causes:

1. A tool that raises ends the run. `create_agent` only turns invalid-argument errors into
   messages for the model and re-raises anything else, where `raw` and `mcp` report every
   failure. Fixed with `ToolErrorMiddleware`, returning the same `Error: ...` text as `raw`.
2. The tool descriptions differed by four spaces. Given a plain function, LangChain keeps the
   docstring's inner indentation; `raw` and FastMCP dedent it:

   ```text
   raw, mcp   "Evaluate an arithmetic expression such as '(4871 * 209) - 17'.\n\nSupports + - ..."
   langchain  "Evaluate an arithmetic expression such as '(4871 * 209) - 17'.\n\n    Supports + - ..."
   ```

   With the indented version, `qwen3.5:4b-mlx` stopped calling `today` and computed
   `2100 - 2024`, a year it guessed. Sending the first request with each combination of the
   three differences found between LangChain's and `raw`'s (streaming, indentation,
   `additionalProperties: false`) isolated the indentation: 12 tries out of 12 with it, none
   without. Why four spaces change that decision is not established. Fixed by passing the
   dedented description to `StructuredTool.from_function`.

`LangChainAgent(framework_defaults=True)` drops both adjustments; the first column comes from
`summary-20261021T121042Z.md` and `summary-20261021T121828Z.md`, the second from
`summary-20261029T110406Z.md` and `summary-20261029T110703Z.md`.

What it shows:

- Once the three variants send the model the same request and report errors the same way, they
  are indistinguishable: equal rates, the same number of model calls, median latencies within
  0.6 s. Neither the in-process MCP round trip nor the LangChain graph shows next to the model's
  own time.
- The framework's defaults are where the variants differed. Out of the box, `langchain` lost 20
  answers of 120 on `llama3.2:3b` and 40 on `qwen3.5:4b-mlx`; each loss traced to something the
  framework did differently, not to the model.
- The model matters more than the machinery: on the same questions, `qwen3.5:4b-mlx` answers all
  120 and `llama3.2:3b` 100, with half its runs on a forbidden or failed path.
- A measured comparison needs identical requests. Before 21 October, `raw` and `mcp` sent no
  temperature at all, which produced the run-to-run variability an earlier version of this
  README described (`summary-20261021T074440Z.md` and `summary-20261021T075135Z.md`, three
  questions, measured after the fix: 360 runs, no failure).

So the framework does not buy capability here. It buys not writing the loop, and it brings
defaults that have to be checked against what the hand-written loop does.

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
Two more check that the LangChain variant binds the same descriptions and reports a failing tool
to the model, and two keep its out-of-the-box behaviour documented (`framework_defaults=True`).

## History

This repo was a tour of LangChain 0.3 written in an object-oriented style. LangChain 1.x removed
most of what it was built on (`initialize_agent`, `AgentType`, `load_tools`, `LLMChain`, the
legacy memories), which made it a choice between redoing the tour on the new API and asking a
question that does not expire with the next major version. The old modules are in the git
history.
