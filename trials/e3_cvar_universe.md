# E3 — CVaR Trader: Universe-Selection Variations

**Window:** 2025-06-10 → 2026-06-10 (252 trading days)  
**Universe selected as of:** 2023-12-24 (no lookahead)

How the universe-selection knobs affect the CVaR trader:

- **uniform / sharpe:** `['GLD', 'AVGO', 'UNH', 'HYG', 'JPM', 'XOM', 'MCD', 'GOOGL']`
- **adaptive / sharpe:** `['AVGO', 'CAT', 'XOM', 'XLE', 'CVX', 'USO', 'JPM', 'MCD']`
- **uniform / calmar:** `['GLD', 'AVGO', 'UNH', 'LQD', 'JPM', 'XOM', 'MCD', 'GOOGL']`


## Results

| Strategy | Ann. Return | Ann. Vol | Sharpe | Max Drawdown |
|---|---|---|---|---|
| CVaR (uniform / sharpe) | 22.68% | 9.62% | 2.36 | -5.38% |
| CVaR (adaptive / sharpe) | 34.08% | 12.31% | 2.77 | -4.56% |
| CVaR (uniform / calmar) | 22.73% | 9.43% | 2.41 | -5.76% |
| S&P 500 (SPY) | 20.29% | 12.23% | 1.66 | -9.13% |
| Dow Jones (DIA) | 16.44% | 12.40% | 1.33 | -10.06% |

![results](e3_cvar_universe.png)
