# E2 — Mean-Variance Trader vs S&P 500 vs Dow Jones

**Window:** 2025-07-24 → 2026-07-24 (252 trading days)  
**Universe selected as of:** 2022-06-17 (no lookahead)

Practical mean-variance trader (Black-Litterman returns + IEWMA covariance + weekly rebalancing + volatility targeting) on a 9-asset PCA/k-means universe.

**Universe:** `['TSLA', 'LQD', 'CAT', 'PDBC', 'KO', 'SLV', 'EEM', 'UNH', 'XLI']`

## Results

| Strategy | Ann. Return | Ann. Vol | Sharpe (excess) | Max Drawdown |
|---|---|---|---|---|
| Mean-Variance (uniform/sharpe, 9) | 42.31% | 14.08% | 2.72 | -7.21% |
| S&P 500 (SPY) | 17.77% | 12.71% | 1.08 | -8.88% |
| Dow Jones (DIA) | 17.77% | 12.26% | 1.12 | -9.76% |
| 60/40 (SPY/IEF) | 11.52% | 8.25% | 0.91 | -6.00% |

![results](e2_mv_baseline.png)
