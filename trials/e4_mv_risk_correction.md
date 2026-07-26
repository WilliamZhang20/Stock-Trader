# E4 — Mean-Variance Practical Upgrade vs Legacy

**Window:** 2025-07-24 → 2026-07-24 (252 trading days)  
**Universe selected as of:** 2022-06-17 (no lookahead)

Ablation of the v6 practical-Markowitz upgrade against the previous defaults. Practical uses Iterated-EWMA covariance, weekly rebalancing, W_MAX=0.25, lambda_risk=12, TARGET_VOL=0.14. Legacy uses Ledoit-Wolf, daily rebalancing, W_MAX=0.40, lambda_risk=7, TARGET_VOL=0.12. Both still use Black-Litterman returns and charge 10 bps of turnover costs.

**Universe:** `['CAT', 'HYG', 'UNH', 'SLV', 'TSLA', 'KO', 'PDBC', 'EEM', 'XLI']`

## Results

| Strategy | Ann. Return | Ann. Vol | Sharpe (excess) | Max Drawdown |
|---|---|---|---|---|
| MV practical | 40.33% | 14.08% | 2.58 | -7.39% |
| MV legacy | 30.28% | 17.88% | 1.47 | -10.60% |
| S&P 500 (SPY) | 17.77% | 12.71% | 1.08 | -8.88% |
| Dow Jones (DIA) | 17.77% | 12.26% | 1.12 | -9.76% |
| 60/40 (SPY/IEF) | 11.52% | 8.25% | 0.91 | -6.00% |

![results](e4_mv_risk_correction.png)
