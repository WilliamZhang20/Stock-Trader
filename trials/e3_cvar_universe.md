# E3 — CVaR Trader: Universe-Selection Variations

**Window:** 2025-07-24 → 2026-07-24 (252 trading days)  
**Universe selected as of:** 2022-06-17 (no lookahead)

How the universe-selection knobs affect the CVaR trader:

- **uniform / sharpe:** `['UNH', 'PDBC', 'TSLA', 'KO', 'CAT', 'LQD', 'EEM', 'SLV']`
- **adaptive / sharpe:** `['EEM', 'UNH', 'CAT', 'PDBC', 'USO', 'XOM', 'XLE', 'TSLA']`
- **uniform / calmar:** `['UNH', 'PDBC', 'TSLA', 'KO', 'CAT', 'LQD', 'EEM', 'SLV']`


## Results

| Strategy | Ann. Return | Ann. Vol | Sharpe (excess) | Max Drawdown |
|---|---|---|---|---|
| CVaR (uniform / sharpe) | 38.43% | 10.82% | 3.18 | -5.06% |
| CVaR (adaptive / sharpe) | 45.28% | 13.01% | 3.17 | -5.84% |
| CVaR (uniform / calmar) | 38.43% | 10.82% | 3.18 | -5.06% |
| S&P 500 (SPY) | 17.77% | 12.71% | 1.08 | -8.88% |
| Dow Jones (DIA) | 17.77% | 12.26% | 1.12 | -9.76% |
| 60/40 (SPY/IEF) | 11.52% | 8.25% | 0.91 | -6.00% |

![results](e3_cvar_universe.png)
