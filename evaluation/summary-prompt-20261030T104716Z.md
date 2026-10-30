# Structured prompt bench

- measured: 20261030T104716Z
- model: `qwen3.5:4b-mlx`, temperature 0.0, think False
- products: 100, runs per product and variant: 1

Rates are successes/products with their 95% Wilson interval. format ok: the answer is exactly one allowed category. category ok: the answer names one allowed category, the right one, whatever the format. errors counts runs that raised or timed out; s median and s p90 are wall-clock seconds per product.

| category | variant | format ok | category ok | errors | s median | s p90 |
|---|---|---|---|---|---|---|
| Electronics | raw | 25/25 (0.87-1.00) | 13/25 (0.33-0.70) | 0 | 0.3 | 0.7 |
| Electronics | langchain | 25/25 (0.87-1.00) | 12/25 (0.30-0.67) | 0 | 0.3 | 0.9 |
| Electronics | mcp | 25/25 (0.87-1.00) | 12/25 (0.30-0.67) | 0 | 0.3 | 0.7 |
| Computers&Accessories | raw | 25/25 (0.87-1.00) | 25/25 (0.87-1.00) | 0 | 0.3 | 0.8 |
| Computers&Accessories | langchain | 25/25 (0.87-1.00) | 25/25 (0.87-1.00) | 0 | 0.3 | 0.8 |
| Computers&Accessories | mcp | 25/25 (0.87-1.00) | 25/25 (0.87-1.00) | 0 | 0.3 | 0.9 |
| Home&Kitchen | raw | 25/25 (0.87-1.00) | 25/25 (0.87-1.00) | 0 | 0.3 | 0.8 |
| Home&Kitchen | langchain | 25/25 (0.87-1.00) | 25/25 (0.87-1.00) | 0 | 0.3 | 0.7 |
| Home&Kitchen | mcp | 25/25 (0.87-1.00) | 25/25 (0.87-1.00) | 0 | 0.3 | 0.6 |
| OfficeProducts | raw | 25/25 (0.87-1.00) | 23/25 (0.75-0.98) | 0 | 0.2 | 0.6 |
| OfficeProducts | langchain | 25/25 (0.87-1.00) | 23/25 (0.75-0.98) | 0 | 0.2 | 0.5 |
| OfficeProducts | mcp | 25/25 (0.87-1.00) | 23/25 (0.75-0.98) | 0 | 0.2 | 0.5 |
| all | raw | 100/100 (0.96-1.00) | 86/100 (0.78-0.91) | 0 | 0.3 | 0.7 |
| all | langchain | 100/100 (0.96-1.00) | 85/100 (0.77-0.91) | 0 | 0.3 | 0.8 |
| all | mcp | 100/100 (0.96-1.00) | 85/100 (0.77-0.91) | 0 | 0.3 | 0.7 |

Confusion, `raw` (rows: true category, columns: category named)

|  | Electronics | Computers&Accessories | Home&Kitchen | OfficeProducts | (none or several) |
|---|---|---|---|---|---|
| Electronics | 13 | 11 | 1 | 0 | 0 |
| Computers&Accessories | 0 | 25 | 0 | 0 | 0 |
| Home&Kitchen | 0 | 0 | 25 | 0 | 0 |
| OfficeProducts | 0 | 2 | 0 | 23 | 0 |

Confusion, `langchain` (rows: true category, columns: category named)

|  | Electronics | Computers&Accessories | Home&Kitchen | OfficeProducts | (none or several) |
|---|---|---|---|---|---|
| Electronics | 12 | 12 | 1 | 0 | 0 |
| Computers&Accessories | 0 | 25 | 0 | 0 | 0 |
| Home&Kitchen | 0 | 0 | 25 | 0 | 0 |
| OfficeProducts | 0 | 2 | 0 | 23 | 0 |

Confusion, `mcp` (rows: true category, columns: category named)

|  | Electronics | Computers&Accessories | Home&Kitchen | OfficeProducts | (none or several) |
|---|---|---|---|---|---|
| Electronics | 12 | 12 | 1 | 0 | 0 |
| Computers&Accessories | 0 | 25 | 0 | 0 | 0 |
| Home&Kitchen | 0 | 0 | 25 | 0 | 0 |
| OfficeProducts | 0 | 2 | 0 | 23 | 0 |
