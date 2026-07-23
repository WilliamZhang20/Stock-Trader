# E2 — Mean-Variance Trader vs S&P 500 vs Dow Jones

**Window:** 2025-07-22 → 2026-07-22 (252 trading days)  
**Universe selected as of:** 2022-06-14 (no lookahead)

Baseline mean-variance trader (Black-Litterman posterior returns + volatility targeting; the old HMM regime model has been removed) on a 9-asset PCA/k-means universe.

**Universe:** `['CAT', 'HYG', 'UNH', 'SLV', 'AVGO', 'KO', 'PDBC', 'EEM', 'TSLA']`

## Results

| Strategy | Ann. Return | Ann. Vol | Sharpe (excess) | Max Drawdown |
|---|---|---|---|---|
| Mean-Variance (uniform/sharpe, 9) | 12.06% | 13.84% | 0.58 | -10.24% |
| S&P 500 (SPY) | 20.17% | 12.66% | 1.28 | -8.88% |
| Dow Jones (DIA) | 18.96% | 12.28% | 1.22 | -9.76% |
| 60/40 (SPY/IEF) | 12.68% | 8.20% | 1.06 | -6.00% |

![results](e2_mv_baseline.png)
