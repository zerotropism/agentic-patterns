# Conversation memory bench

- measured: 20261031T160544Z
- model: `llama3.2:3b`, temperature 0.0, think False
- conversations: 36, runs per conversation and variant: 1

distance is the number of turns between the fact and the question. right and distractor are conversations/runs with their 95% Wilson interval: the last answer names the fact only, or the distractor only. errors counts runs that raised or timed out. last prompt tokens is the median size of the last prompt, as Ollama counted it; s median and s p90 are wall-clock seconds per conversation, every turn included.

| distance | variant | right | distractor | errors | last prompt tokens | s median | s p90 |
|---|---|---|---|---|---|---|---|
| 1 | raw | 12/12 (0.76-1.00) | 0/12 (0.00-0.24) | 0 | 161 | 2.3 | 2.5 |
| 1 | langchain | 12/12 (0.76-1.00) | 0/12 (0.00-0.24) | 0 | 161 | 2.3 | 2.5 |
| 1 | mcp | 12/12 (0.76-1.00) | 0/12 (0.00-0.24) | 0 | 161 | 2.3 | 2.6 |
| 3 | raw | 12/12 (0.76-1.00) | 0/12 (0.00-0.24) | 0 | 230 | 3.2 | 3.8 |
| 3 | langchain | 12/12 (0.76-1.00) | 0/12 (0.00-0.24) | 0 | 230 | 3.3 | 3.9 |
| 3 | mcp | 12/12 (0.76-1.00) | 0/12 (0.00-0.24) | 0 | 230 | 3.4 | 3.8 |
| 6 | raw | 12/12 (0.76-1.00) | 0/12 (0.00-0.24) | 0 | 330 | 4.6 | 5.3 |
| 6 | langchain | 12/12 (0.76-1.00) | 0/12 (0.00-0.24) | 0 | 330 | 4.6 | 5.5 |
| 6 | mcp | 12/12 (0.76-1.00) | 0/12 (0.00-0.24) | 0 | 330 | 4.7 | 5.5 |
| all | raw | 36/36 (0.90-1.00) | 0/36 (0.00-0.10) | 0 | 230 | 3.2 | 4.9 |
| all | langchain | 36/36 (0.90-1.00) | 0/36 (0.00-0.10) | 0 | 230 | 3.3 | 4.9 |
| all | mcp | 36/36 (0.90-1.00) | 0/36 (0.00-0.10) | 0 | 230 | 3.4 | 4.9 |
