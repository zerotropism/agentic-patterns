# Typed parsing bench

- measured: 20261031T151143Z
- model: `llama3.2:3b`, temperature 0.0, think False
- listings: 100, runs per listing and variant: 1

valid and all fields are successes/listings with their 95% Wilson interval; the five field columns count listings whose field is right. An invalid answer counts as wrong on every field. errors counts runs that raised or timed out; s median and s p90 are wall-clock seconds per listing.

| phrasing | variant | valid | all fields | actual | discounted | discount | rating | count | errors | s median | s p90 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| was_now | raw | 25/25 (0.87-1.00) | 21/25 (0.65-0.94) | 24/25 | 25/25 | 25/25 | 25/25 | 22/25 | 0 | 1.1 | 1.2 |
| was_now | langchain | 25/25 (0.87-1.00) | 21/25 (0.65-0.94) | 24/25 | 25/25 | 25/25 | 25/25 | 22/25 | 0 | 1.0 | 1.0 |
| deal | raw | 25/25 (0.87-1.00) | 23/25 (0.75-0.98) | 24/25 | 25/25 | 25/25 | 25/25 | 24/25 | 0 | 1.0 | 1.1 |
| deal | langchain | 25/25 (0.87-1.00) | 23/25 (0.75-0.98) | 24/25 | 25/25 | 25/25 | 25/25 | 24/25 | 0 | 1.2 | 1.2 |
| plain | raw | 25/25 (0.87-1.00) | 25/25 (0.87-1.00) | 25/25 | 25/25 | 25/25 | 25/25 | 25/25 | 0 | 1.1 | 1.2 |
| plain | langchain | 25/25 (0.87-1.00) | 25/25 (0.87-1.00) | 25/25 | 25/25 | 25/25 | 25/25 | 25/25 | 0 | 1.0 | 1.1 |
| sheet | raw | 25/25 (0.87-1.00) | 25/25 (0.87-1.00) | 25/25 | 25/25 | 25/25 | 25/25 | 25/25 | 0 | 1.0 | 1.1 |
| sheet | langchain | 25/25 (0.87-1.00) | 25/25 (0.87-1.00) | 25/25 | 25/25 | 25/25 | 25/25 | 25/25 | 0 | 1.2 | 1.2 |
| all | raw | 100/100 (0.96-1.00) | 94/100 (0.88-0.97) | 98/100 | 100/100 | 100/100 | 100/100 | 96/100 | 0 | 1.1 | 1.2 |
| all | langchain | 100/100 (0.96-1.00) | 94/100 (0.88-0.97) | 98/100 | 100/100 | 100/100 | 100/100 | 96/100 | 0 | 1.1 | 1.2 |
