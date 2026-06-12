# E1 — CVaR Trader vs S&P 500 vs Dow Jones

**Window:** 2025-06-10 → 2026-06-10 (252 trading days)  
**Universe selected as of:** 2023-12-24 (no lookahead)

Baseline CVaR trader on an 8-asset universe selected by PCA + k-means (one asset per cluster, Sharpe-ranked).

**Universe:** `['GLD', 'AVGO', 'UNH', 'HYG', 'JPM', 'XOM', 'MCD', 'GOOGL']`

## Results

| Strategy | Ann. Return | Ann. Vol | Sharpe | Max Drawdown |
|---|---|---|---|---|
| CVaR (uniform/sharpe, 8) | 22.68% | 9.62% | 2.36 | -5.38% |
| S&P 500 (SPY) | 20.29% | 12.23% | 1.66 | -9.13% |
| Dow Jones (DIA) | 16.44% | 12.40% | 1.33 | -10.06% |

![results](e1_cvar_baseline.png)
