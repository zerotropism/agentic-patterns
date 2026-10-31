# Typed parsing bench

- measured: 20261031T151528Z
- model: `qwen3.5:4b-mlx`, temperature 0.0, think False
- listings: 100, runs per listing and variant: 1

valid and all fields are successes/listings with their 95% Wilson interval; the five field columns count listings whose field is right. An invalid answer counts as wrong on every field. errors counts runs that raised or timed out; s median and s p90 are wall-clock seconds per listing.

| phrasing | variant | valid | all fields | actual | discounted | discount | rating | count | errors | s median | s p90 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| was_now | raw | 25/25 (0.87-1.00) | 25/25 (0.87-1.00) | 25/25 | 25/25 | 25/25 | 25/25 | 25/25 | 0 | 2.6 | 2.7 |
| was_now | langchain | 25/25 (0.87-1.00) | 25/25 (0.87-1.00) | 25/25 | 25/25 | 25/25 | 25/25 | 25/25 | 0 | 2.4 | 2.5 |
| deal | raw | 25/25 (0.87-1.00) | 25/25 (0.87-1.00) | 25/25 | 25/25 | 25/25 | 25/25 | 25/25 | 0 | 2.4 | 2.6 |
| deal | langchain | 25/25 (0.87-1.00) | 25/25 (0.87-1.00) | 25/25 | 25/25 | 25/25 | 25/25 | 25/25 | 0 | 2.6 | 2.8 |
| plain | raw | 25/25 (0.87-1.00) | 25/25 (0.87-1.00) | 25/25 | 25/25 | 25/25 | 25/25 | 25/25 | 0 | 2.6 | 2.8 |
| plain | langchain | 25/25 (0.87-1.00) | 25/25 (0.87-1.00) | 25/25 | 25/25 | 25/25 | 25/25 | 25/25 | 0 | 2.4 | 2.5 |
| sheet | raw | 25/25 (0.87-1.00) | 25/25 (0.87-1.00) | 25/25 | 25/25 | 25/25 | 25/25 | 25/25 | 0 | 2.4 | 2.5 |
| sheet | langchain | 25/25 (0.87-1.00) | 25/25 (0.87-1.00) | 25/25 | 25/25 | 25/25 | 25/25 | 25/25 | 0 | 2.6 | 2.7 |
| all | raw | 100/100 (0.96-1.00) | 100/100 (0.96-1.00) | 100/100 | 100/100 | 100/100 | 100/100 | 100/100 | 0 | 2.5 | 2.7 |
| all | langchain | 100/100 (0.96-1.00) | 100/100 (0.96-1.00) | 100/100 | 100/100 | 100/100 | 100/100 | 100/100 | 0 | 2.5 | 2.7 |
