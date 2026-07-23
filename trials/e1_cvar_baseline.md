# E1 — CVaR Trader vs S&P 500 vs Dow Jones

**Window:** 2025-07-22 → 2026-07-22 (252 trading days)  
**Universe selected as of:** 2022-06-14 (no lookahead)

Baseline CVaR trader on an 8-asset universe selected by PCA + k-means (one asset per cluster, Sharpe-ranked).

**Universe:** `['KO', 'TSLA', 'CAT', 'EEM', 'LQD', 'SLV', 'PDBC', 'UNH']`

## Results

| Strategy | Ann. Return | Ann. Vol | Sharpe (excess) | Max Drawdown |
|---|---|---|---|---|
| CVaR (uniform/sharpe, 8) | 38.90% | 10.81% | 3.23 | -5.06% |
| S&P 500 (SPY) | 20.17% | 12.66% | 1.28 | -8.88% |
| Dow Jones (DIA) | 18.96% | 12.28% | 1.22 | -9.76% |
| 60/40 (SPY/IEF) | 12.68% | 8.20% | 1.06 | -6.00% |

![results](e1_cvar_baseline.png)
