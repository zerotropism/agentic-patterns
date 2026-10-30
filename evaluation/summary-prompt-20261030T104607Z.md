# Structured prompt bench

- measured: 20261030T104607Z
- model: `llama3.2:3b`, temperature 0.0, think False
- products: 100, runs per product and variant: 1

Rates are successes/products with their 95% Wilson interval. format ok: the answer is exactly one allowed category. category ok: the answer names one allowed category, the right one, whatever the format. errors counts runs that raised or timed out; s median and s p90 are wall-clock seconds per product.

| category | variant | format ok | category ok | errors | s median | s p90 |
|---|---|---|---|---|---|---|
| Electronics | raw | 25/25 (0.87-1.00) | 19/25 (0.57-0.89) | 0 | 0.1 | 0.5 |
| Electronics | langchain | 25/25 (0.87-1.00) | 19/25 (0.57-0.89) | 0 | 0.1 | 0.5 |
| Electronics | mcp | 25/25 (0.87-1.00) | 19/25 (0.57-0.89) | 0 | 0.1 | 0.6 |
| Computers&Accessories | raw | 25/25 (0.87-1.00) | 8/25 (0.17-0.52) | 0 | 0.2 | 0.5 |
| Computers&Accessories | langchain | 25/25 (0.87-1.00) | 8/25 (0.17-0.52) | 0 | 0.2 | 0.5 |
| Computers&Accessories | mcp | 25/25 (0.87-1.00) | 8/25 (0.17-0.52) | 0 | 0.2 | 0.6 |
| Home&Kitchen | raw | 25/25 (0.87-1.00) | 19/25 (0.57-0.89) | 0 | 0.1 | 0.5 |
| Home&Kitchen | langchain | 25/25 (0.87-1.00) | 19/25 (0.57-0.89) | 0 | 0.1 | 0.5 |
| Home&Kitchen | mcp | 25/25 (0.87-1.00) | 19/25 (0.57-0.89) | 0 | 0.1 | 0.5 |
| OfficeProducts | raw | 25/25 (0.87-1.00) | 21/25 (0.65-0.94) | 0 | 0.1 | 0.4 |
| OfficeProducts | langchain | 25/25 (0.87-1.00) | 21/25 (0.65-0.94) | 0 | 0.1 | 0.3 |
| OfficeProducts | mcp | 25/25 (0.87-1.00) | 21/25 (0.65-0.94) | 0 | 0.1 | 0.4 |
| all | raw | 100/100 (0.96-1.00) | 67/100 (0.57-0.75) | 0 | 0.1 | 0.5 |
| all | langchain | 100/100 (0.96-1.00) | 67/100 (0.57-0.75) | 0 | 0.1 | 0.5 |
| all | mcp | 100/100 (0.96-1.00) | 67/100 (0.57-0.75) | 0 | 0.1 | 0.5 |

Confusion, `raw` (rows: true category, columns: category named)

|  | Electronics | Computers&Accessories | Home&Kitchen | OfficeProducts | (none or several) |
|---|---|---|---|---|---|
| Electronics | 19 | 1 | 4 | 1 | 0 |
| Computers&Accessories | 13 | 8 | 2 | 2 | 0 |
| Home&Kitchen | 5 | 0 | 19 | 1 | 0 |
| OfficeProducts | 0 | 0 | 4 | 21 | 0 |

Confusion, `langchain` (rows: true category, columns: category named)

|  | Electronics | Computers&Accessories | Home&Kitchen | OfficeProducts | (none or several) |
|---|---|---|---|---|---|
| Electronics | 19 | 1 | 4 | 1 | 0 |
| Computers&Accessories | 13 | 8 | 2 | 2 | 0 |
| Home&Kitchen | 5 | 0 | 19 | 1 | 0 |
| OfficeProducts | 0 | 0 | 4 | 21 | 0 |

Confusion, `mcp` (rows: true category, columns: category named)

|  | Electronics | Computers&Accessories | Home&Kitchen | OfficeProducts | (none or several) |
|---|---|---|---|---|---|
| Electronics | 19 | 1 | 4 | 1 | 0 |
| Computers&Accessories | 13 | 8 | 2 | 2 | 0 |
| Home&Kitchen | 5 | 0 | 19 | 1 | 0 |
| OfficeProducts | 0 | 0 | 4 | 21 | 0 |
