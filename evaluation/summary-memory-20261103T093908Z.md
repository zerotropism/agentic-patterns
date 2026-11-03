# Conversation memory bench

- measured: 20261103T093908Z
- model: `qwen3.5:4b-mlx`, temperature 0.0, think False
- conversations: 36, runs per conversation and variant: 1

distance is the number of turns between the fact and the question. right and distractor are conversations/runs with their 95% Wilson interval: the last answer names the fact only, or the distractor only. errors counts runs that raised or timed out. last prompt tokens is the median size of the last prompt, as Ollama counted it; s median and s p90 are wall-clock seconds per conversation, every turn included.

| distance | variant | right | distractor | errors | last prompt tokens | s median | s p90 |
|---|---|---|---|---|---|---|---|
| 1 | raw | 12/12 (0.76-1.00) | 0/12 (0.00-0.24) | 0 | 142 | 3.7 | 5.0 |
| 1 | langchain | 12/12 (0.76-1.00) | 0/12 (0.00-0.24) | 0 | 140 | 3.6 | 5.1 |
| 1 | mcp | 11/12 (0.65-0.99) | 0/12 (0.00-0.24) | 0 | 140 | 3.7 | 4.9 |
| 3 | raw | 10/12 (0.55-0.95) | 0/12 (0.00-0.24) | 0 | 206 | 4.7 | 5.8 |
| 3 | langchain | 10/12 (0.55-0.95) | 0/12 (0.00-0.24) | 0 | 206 | 4.9 | 5.9 |
| 3 | mcp | 10/12 (0.55-0.95) | 0/12 (0.00-0.24) | 0 | 206 | 4.8 | 5.9 |
| 6 | raw | 10/12 (0.55-0.95) | 0/12 (0.00-0.24) | 0 | 310 | 7.6 | 8.9 |
| 6 | langchain | 10/12 (0.55-0.95) | 0/12 (0.00-0.24) | 0 | 310 | 7.4 | 8.9 |
| 6 | mcp | 10/12 (0.55-0.95) | 0/12 (0.00-0.24) | 0 | 310 | 7.7 | 8.9 |
| all | raw | 32/36 (0.75-0.96) | 0/36 (0.00-0.10) | 0 | 206 | 4.9 | 8.0 |
| all | langchain | 32/36 (0.75-0.96) | 0/36 (0.00-0.10) | 0 | 206 | 5.3 | 8.5 |
| all | mcp | 31/36 (0.71-0.94) | 0/36 (0.00-0.10) | 0 | 206 | 5.1 | 8.3 |
