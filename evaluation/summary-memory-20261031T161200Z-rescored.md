# Conversation memory bench

- measured: 20261031T161200Z
- model: `qwen3.5:4b-mlx`, temperature 0.0, think False
- conversations: 36, runs per conversation and variant: 1

distance is the number of turns between the fact and the question. right and distractor are conversations/runs with their 95% Wilson interval: the last answer names the fact only, or the distractor only. errors counts runs that raised or timed out. last prompt tokens is the median size of the last prompt, as Ollama counted it; s median and s p90 are wall-clock seconds per conversation, every turn included.

| distance | variant | right | distractor | errors | last prompt tokens | s median | s p90 |
|---|---|---|---|---|---|---|---|
| 1 | raw | 12/12 (0.76-1.00) | 0/12 (0.00-0.24) | 0 | 143 | 3.5 | 4.7 |
| 1 | langchain | 11/12 (0.65-0.99) | 0/12 (0.00-0.24) | 0 | 142 | 3.3 | 4.7 |
| 1 | mcp | 12/12 (0.76-1.00) | 0/12 (0.00-0.24) | 0 | 142 | 3.3 | 4.6 |
| 3 | raw | 11/12 (0.65-0.99) | 0/12 (0.00-0.24) | 0 | 204 | 4.5 | 5.2 |
| 3 | langchain | 11/12 (0.65-0.99) | 0/12 (0.00-0.24) | 0 | 204 | 4.6 | 5.6 |
| 3 | mcp | 10/12 (0.55-0.95) | 0/12 (0.00-0.24) | 0 | 206 | 4.5 | 6.0 |
| 6 | raw | 11/12 (0.65-0.99) | 0/12 (0.00-0.24) | 0 | 310 | 7.0 | 9.2 |
| 6 | langchain | 9/12 (0.47-0.91) | 0/12 (0.00-0.24) | 0 | 313 | 7.1 | 8.3 |
| 6 | mcp | 10/12 (0.55-0.95) | 0/12 (0.00-0.24) | 0 | 310 | 7.6 | 8.5 |
| all | raw | 34/36 (0.82-0.98) | 0/36 (0.00-0.10) | 0 | 204 | 4.6 | 7.9 |
| all | langchain | 31/36 (0.71-0.94) | 0/36 (0.00-0.10) | 0 | 204 | 4.8 | 8.2 |
| all | mcp | 32/36 (0.75-0.96) | 0/36 (0.00-0.10) | 0 | 206 | 4.6 | 8.1 |
