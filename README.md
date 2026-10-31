# agentic-patterns

The same agentic use cases, each implemented three ways, side by side: by hand with the Ollama
client, with LangChain 1.x, and with an MCP server. Four cases so far: an agent with tools, a
structured prompt, typed parsing and conversation memory. Each one is measured by the same
bench.

The point is not which one wins. It is what you have to write, what the framework writes for
you, and whether that changes the outcome.

## Try it

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/) and a running [Ollama](https://ollama.com/).

```bash
uv sync
ollama pull llama3.2:3b

uv run agentic-patterns tools                     # what the three agent variants share
uv run agentic-patterns run combined --variant raw
uv run agentic-patterns bench                     # agent: every variant, every question, 20 runs
uv run agentic-patterns classify --variant mcp    # structured prompt: one product
uv run agentic-patterns bench --case prompt       # structured prompt: 100 products
uv run agentic-patterns extract --variant mcp     # typed parsing: one listing
uv run agentic-patterns bench --case parsing      # typed parsing: 100 listings
uv run agentic-patterns converse --variant mcp    # memory: one conversation
uv run agentic-patterns bench --case memory       # memory: 36 conversations
```

The model comes from `config.yaml` and can be overridden per run:

```bash
AGENTIC_PATTERNS_MODEL=qwen3.5:4b-mlx uv run agentic-patterns bench
```

## Agent with tools

### The three variants

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

### What the tools are

Two plain Python functions, shared by every variant: `today()` and `calculate(expression)`.
`calculate` evaluates arithmetic through an AST walk that accepts numbers and operators and
nothing else — no names, no calls, no attributes. It replaces both the `llm-math` chain, which
had the model do the arithmetic, and `PythonREPLTool`, which executed whatever the model wrote.

Their docstrings are the descriptions sent to the model, in all three variants. One place
describes each tool.

### Bench

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
is why the two counts are kept apart. Variants are interleaved, their order rotates from one
question and run to the next, and one warm-up run per variant is discarded. A run that raises or exceeds the
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

## Structured prompt

### The task

Put a product of an online shop in one of four categories: Electronics, Computers&Accessories,
Home&Kitchen, OfficeProducts. The products come from `data/amazon.csv`, kept from the LangChain
course this repo started as: the model gets the name and the first 500 characters of the
description, and the reference is the first segment of the category path in the file
(`Computers&Accessories|Cables|USBCables` is Computers&Accessories). The file repeats a product
once per review; 1,351 distinct products remain.

The prompt is the same text with `{placeholders}` for all three variants, in `config.yaml`:
instructions listing the categories, one example exchange per category (the first product of
each category in the file), then the product. The bench draws 25 other products per category
with a fixed seed, 100 in all, none of them shown as an example.

### The three variants

| | template | rendering | where the template lives |
|---|---|---|---|
| `raw` | text from `config.yaml` | `str.format`, messages built by hand | the client |
| `langchain` | the same text | `ChatPromptTemplate` with a `FewShotChatMessagePromptTemplate` | the client |
| `mcp` | the same text | a FastMCP prompt, `classify_product(name, description)`, fetched with `get_prompt` | the server |

What each one brings:

- `raw` checks nothing: a placeholder without a value raises `KeyError` at the first call, a
  value without a placeholder is silently dropped.
- `langchain` declares the template's variables, so a missing one is reported before the model
  is called, and builds the few-shot exchanges from a list of examples.
- `mcp` moves the template to the server: a client needs only the prompt's name and its two
  string arguments, both declared as required, and the template can change without any client
  changing. MCP prompts have user and assistant messages only, no system role. So that the three
  variants send the same request, the instructions are the first user message in all three.
  The server renders with `raw`'s code; what MCP changes is where it runs.

Tests check that the three variants produce the same messages, that `raw` and `mcp` send Ollama
the same request, and that `langchain` sends the same roles, contents, temperature and `think`.

### Bench

`agentic-patterns bench --case prompt` classifies each product once per variant, and scores two
things apart:

- format: the answer is exactly one allowed category, nothing before or after;
- category: the answer names exactly one allowed category, and it is the right one. `Category:
  Home & Kitchen.` fails the format and passes the category.

A wrong format is fixed by parsing; a wrong category is not. The summary adds one confusion
matrix per variant. Files: `evaluation/outcomes-prompt-*.jsonl` and `summary-prompt-*.md`.

### Results

Measured on 30 October 2026, temperature 0, `think: false`, 100 products, one run per product
and variant (`summary-prompt-20261030T104607Z.md` and `summary-prompt-20261030T104716Z.md`):

| model | variant | format ok | category ok | s median | s p90 |
|---|---|---|---|---|---|
| `llama3.2:3b` | raw | 100/100 | 67/100 | 0.1 | 0.5 |
| `llama3.2:3b` | langchain | 100/100 | 67/100 | 0.1 | 0.5 |
| `llama3.2:3b` | mcp | 100/100 | 67/100 | 0.1 | 0.5 |
| `qwen3.5:4b-mlx` | raw | 100/100 | 86/100 | 0.3 | 0.7 |
| `qwen3.5:4b-mlx` | langchain | 100/100 | 85/100 | 0.3 | 0.8 |
| `qwen3.5:4b-mlx` | mcp | 100/100 | 85/100 | 0.3 | 0.7 |

Where the errors are, true category in rows, category named in columns, the same for the three
variants except one product:

| `llama3.2:3b` | Electronics | Computers&Accessories | Home&Kitchen | OfficeProducts |
|---|---|---|---|---|
| Electronics | 19 | 1 | 4 | 1 |
| Computers&Accessories | 13 | 8 | 2 | 2 |
| Home&Kitchen | 5 | 0 | 19 | 1 |
| OfficeProducts | 0 | 0 | 4 | 21 |

| `qwen3.5:4b-mlx` | Electronics | Computers&Accessories | Home&Kitchen | OfficeProducts |
|---|---|---|---|---|
| Electronics | 12 or 13 | 11 or 12 | 1 | 0 |
| Computers&Accessories | 0 | 25 | 0 | 0 |
| Home&Kitchen | 0 | 0 | 25 | 0 |
| OfficeProducts | 0 | 2 | 0 | 23 |

What it shows:

- The three variants send the same request and get the same answers: identical on all 100
  products with `llama3.2:3b`; with `qwen3.5:4b-mlx`, one Electronics product named Electronics
  by `raw` and Computers&Accessories by the other two. Their requests are identical by test, so
  that one product is the model not being strictly deterministic at temperature 0, as one run in
  360 was in the agent bench.
- The format is solved by the examples: no answer of 600 had anything around the category.
- The errors sit on one boundary, Electronics against Computers&Accessories, in opposite
  directions: `llama3.2:3b` calls half the computer accessories electronics, `qwen3.5:4b-mlx`
  calls half the electronics computer accessories. The shop's own split is loose there: among
  products named so, 178 cables are Computers&Accessories and 44 Electronics, chargers 28 and 33,
  headphones and power banks fall on both sides too. Part of the error is the reference.

So, for a structured prompt, the choice between the three is not about the answers: they are
the same. It is about what checks the template (`langchain`), and where it lives (`mcp`).

## Typed parsing

### The task

Read the five figures of a product listing into a typed object:

```python
class Listing(BaseModel):
    actual_price: float  # price before discount, in rupees
    discounted_price: float  # price after discount
    discount_percent: int  # 0 to 100
    rating: float  # 0 to 5
    rating_count: int  # 0 or more
```

The field names are the columns of `data/amazon.csv`; each field carries a description. The
listings are written from those columns, so every field has an exact reference: 100 products
drawn with a fixed seed, in four phrasings taken in turn, 25 each.

| phrasing | example |
|---|---|
| `was_now` | `… Was ₹5,890, now ₹2,599 (56% off). Rated 4.1 out of 5 by 21,783 customers.` |
| `deal` | `Deal of the day: … for just ₹664 instead of ₹1,490, that is 55% less. 925 ratings, 4.1/5 on average.` |
| `plain` | `… \| List price: Rs 999 \| Offer price: Rs 379 \| You save 62 percent \| 4.3 stars from 3096 reviews` |
| `sheet` | `Product: …` then `MRP: INR 2,995`, `Selling price: INR 1,625`, `Discount: 46%`, `Rating: 4.5 (23,484 ratings)`, one per line |

### The three variants

| | how the type is enforced | who validates |
|---|---|---|
| `raw` | Ollama's `format=` with the JSON schema of `Listing` | `Listing.model_validate_json`, by hand |
| `langchain` | `with_structured_output(Listing)`; its default method, `json_schema`, sends the same `format=` request as `raw` | the framework |
| `mcp` | a tool, `record_listing`, whose parameters are the fields of `Listing`, with the same types, bounds and descriptions | FastMCP, before the tool runs: a rating of 7 is refused |

One model call per listing in all three, with no second chance after a validation error. An
answer that does not validate counts as wrong on every field.

### Results

Measured on 30 and 31 October 2026, temperature 0, `think: false`, 100 listings, one run per
listing and variant:

| model | variant | valid | all fields | actual | discounted | discount | rating | count |
|---|---|---|---|---|---|---|---|---|
| `llama3.2:3b` | raw | 100 | 94 | 98 | 100 | 100 | 100 | 96 |
| `llama3.2:3b` | langchain | 100 | 94 | 98 | 100 | 100 | 100 | 96 |
| `llama3.2:3b` | mcp | 100 | 95 | 98 | 100 | 100 | 100 | 97 |
| `qwen3.5:4b-mlx` | raw | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| `qwen3.5:4b-mlx` | langchain | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| `qwen3.5:4b-mlx` | mcp | 100 | 100 | 100 | 100 | 100 | 100 | 100 |

`raw` and `langchain` from `summary-parsing-20261031T151143Z.md` and `…T151528Z.md`, `mcp` from
`summary-parsing-20261030T160723Z.md` and `…T161333Z.md`; the `mcp` variant did not change
between the two days.

### What `raw` and `langchain` needed

The first measurement, on 30 October, sent the instructions alone:

| model | variant | all fields | actual | discounted |
|---|---|---|---|---|
| `llama3.2:3b` | raw | 89 | 92 | 99 |
| `llama3.2:3b` | langchain | 90 | 93 | 99 |
| `qwen3.5:4b-mlx` | raw | 3 | 3 | 84 |
| `qwen3.5:4b-mlx` | langchain | 3 | 3 | 84 |

`qwen3.5:4b-mlx` put the price actually paid in `actual_price`: in 83 answers of 100 the two
prices were equal, and in the `deal` phrasing, which gives the discounted price first, it
swapped them. The same model, given the same schema as a tool, got every field right.

`format=` constrains what the model writes to the schema; it does not show the schema to the
model, which sees each key only as it is made to write it, and never its description. A tool's
schema, descriptions included, is part of the prompt. The same 20 listings, sent four ways:

| condition | `qwen3.5:4b-mlx`, actual right | `llama3.2:3b`, actual right |
|---|---|---|
| `format=`, instructions alone | 0/20 | 19/20 |
| `format=`, the schema written after the instructions | 20/20 | 19/20 |
| `format=`, fields renamed `list_price` and `offer_price` | 20/20 | 19/20 |
| a tool with the same schema | 20/20 | 19/20 |

Writing the schema in the prompt, or naming the fields so that their name says it all, each fixes
it. `raw` and `langchain` now write the schema after the instructions (`with_schema` in
`patterns/parsing/task.py`); `schema_in_prompt: false` in `config.yaml` reproduces the first
measurement. `with_structured_output` does not do this for you either: it builds the `format=`
request, not the prompt.

What it shows:

- Once the model sees the schema, the three ways of getting a typed object give the same
  answers: 100/100 for `qwen3.5:4b-mlx`, 94 or 95 for `llama3.2:3b`, whose remaining errors are
  mostly the number of ratings.
- All 1,000 answers of the two measurements validated against the type, in all three variants.
  Constrained output and argument validation both guarantee the shape; neither guarantees the
  meaning.
- A field's description only helps if it reaches the model. With `format=`, it does not, and the
  field's name is all there is.

## Conversation memory

### The task

Answer a question about something said turns earlier. Each conversation states a fact at its
first turn, goes on with unrelated questions, states the same kind of fact about someone else
somewhere in between, and asks for the fact at its last turn. The model answers every turn;
only the last answer is scored.

```text
user: Please note that I live in Kochi.
user: What is the capital of Japan?
user: My sister lives in Pune, a long way from me.
user: Name a fruit that is yellow when ripe.
user: In which city do I live?             -> Kochi, not Pune
```

36 conversations, generated with a fixed seed: 12 at each distance (1, 3 and 6 turns between
the fact and the question), over four kinds of fact taken in turn: a budget, a city, a dog's
name, a delivery day.

### The three variants

| | where the history lives | what each turn sends |
|---|---|---|
| `raw` | a list, local to the run | the whole history, rebuilt by hand |
| `langchain` | a LangGraph checkpointer (`InMemorySaver`), one thread per run, behind `create_agent` | the new message only; the checkpointer puts the history back |
| `mcp` | a FastMCP server: a tool `append_message(session, role, content)` and a resource `history://{session}` | the history read from the resource; the answer is appended after |

The checkpointer replaces the 0.x memories (`ConversationBufferMemory` and its siblings), which
LangChain 1.x removed. The checkpointer and the MCP server outlive a run, so each run gets its
own thread or session; a test checks that two runs do not share a history. Another checks that
the three variants send the model the same messages at every turn.

### Bench

`agentic-patterns bench --case memory` plays each conversation once per variant and scores the
last answer:

- right: it names the fact, and not the distractor;
- distractor: it names the distractor, and not the fact.

An answer naming both, or neither, is neither. Amounts are compared without separators or `₹`,
names as whole words, a plural `s` allowed. The summary gives both rates by distance, the size
of the last prompt as Ollama counted it, and the time per conversation, every turn included.

### Results

Measured on 31 October 2026, temperature 0, `think: false`, one run per conversation and
variant (`summary-memory-20261031T160544Z-rescored.md` and `…T161200Z-rescored.md`):

| model | variant | distance 1 | distance 3 | distance 6 | all | distractor | last prompt, tokens | s median |
|---|---|---|---|---|---|---|---|---|
| `llama3.2:3b` | raw | 12/12 | 12/12 | 12/12 | 36/36 | 0 | 161, 230, 330 | 3.2 |
| `llama3.2:3b` | langchain | 12/12 | 12/12 | 12/12 | 36/36 | 0 | 161, 230, 330 | 3.3 |
| `llama3.2:3b` | mcp | 12/12 | 12/12 | 12/12 | 36/36 | 0 | 161, 230, 330 | 3.4 |
| `qwen3.5:4b-mlx` | raw | 12/12 | 11/12 | 11/12 | 34/36 | 0 | 143, 204, 310 | 4.6 |
| `qwen3.5:4b-mlx` | langchain | 11/12 | 11/12 | 9/12 | 31/36 | 0 | 142, 204, 313 | 4.8 |
| `qwen3.5:4b-mlx` | mcp | 12/12 | 10/12 | 10/12 | 32/36 | 0 | 142, 206, 310 | 4.6 |

The first scoring missed a plural: "deliveries on Tuesdays" did not count as Tuesday. Scored
again with `agentic-patterns rescore`, from the answers already stored, without calling a
model; the first summaries are kept next to the rescored ones.

What the misses are, all from `qwen3.5:4b-mlx`:

- several days, the right one among them: "Tuesday, Wednesday, and Friday", 5 answers;
- no answer: "Your dog's name isn't mentioned in our conversation yet", "I don't know where
  you live", 6 answers.

What it shows:

- No answer of the 216 gave the distractor alone. The history always reached the model, and the
  model never took someone else's fact for the user's.
- The variants differ only through the model. The size of `qwen3.5:4b-mlx`'s last prompt varies
  by one to three tokens from one variant to another: one of its earlier answers was not word
  for word the same, and the rest of the conversation inherited it. The code sends what it is
  given, by test; at temperature 0, the model does not always give the same. With
  `llama3.2:3b`, whose answers were the same, the three variants are identical down to the
  token.
- Distance does not matter at this size: from 140 to 330 tokens, memory is not what limits
  either model.

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
├── patterns/prompt/
│   ├── task.py            products, sample, the shared template
│   ├── raw.py             messages built with str.format
│   ├── with_langchain.py  ChatPromptTemplate, few-shot template
│   ├── with_mcp.py        FastMCP prompt, fetched with get_prompt
│   └── scoring.py         format, category, confusion
├── patterns/parsing/
│   ├── task.py            Listing, the four phrasings, the schema in the prompt
│   ├── raw.py             format= and model_validate_json
│   ├── with_langchain.py  with_structured_output
│   ├── with_mcp.py        a tool whose parameters are the fields
│   └── scoring.py         validity, field by field
├── patterns/memory/
│   ├── task.py            the conversations, the fact, the distractor
│   ├── raw.py             a list, sent whole at every turn
│   ├── with_langchain.py  create_agent and a LangGraph checkpointer
│   ├── with_mcp.py        a FastMCP server holding the history
│   └── scoring.py         right, distractor, by distance
├── cases.py          the use cases the bench can run
├── bench.py          the interleaved run loop, agent scoring, Wilson intervals
├── rendering.py      tables
└── cli.py            cyclopts commands
evaluation/           bench outcomes and summaries, one pair per measurement
```

Adding a variant means adding a class with a `name` and an `async run`, plus one line in its
case's `__init__.py`. Adding a use case means a sibling package under `patterns/`, with its own
scoring, and one entry in `cases.py`; the bench command and its loop do not change.

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
For the structured prompt, the three variants are checked to build the same messages and send
the same request, and the scoring to keep format and category apart. For typed parsing, `raw`
and `langchain` are checked to send the same messages and schema, and the MCP tool's schema to
match `Listing` field by field. For conversation memory, the three variants are checked to send
the same history at every turn, and two runs never to share one.

`agentic-patterns rescore <outcomes> --case <case>` scores a stored run again after a change in
a scorer. The agent case is excluded: its expected date is the day of the run.

## History

This repo was a tour of LangChain 0.3 written in an object-oriented style. LangChain 1.x removed
most of what it was built on (`initialize_agent`, `AgentType`, `load_tools`, `LLMChain`, the
legacy memories), which made it a choice between redoing the tour on the new API and asking a
question that does not expire with the next major version. The old modules are in the git
history.
