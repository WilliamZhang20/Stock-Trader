# E4 — Mean-Variance Risk-Tightening Correction

**Window:** 2025-06-10 → 2026-06-10 (252 trading days)  
**Universe selected as of:** 2023-12-24 (no lookahead)

A simple, pragmatic correction to the mean-variance objective/constraints: halve the per-asset weight cap (0.40 → 0.20) and roughly double risk aversion (7 → 15). This trades some upside for materially smaller drawdowns.

**Universe:** `['EEM', 'AVGO', 'CAT', 'UNH', 'HYG', 'GLD', 'JPM', 'XOM', 'MCD']`

## Results

| Strategy | Ann. Return | Ann. Vol | Sharpe | Max Drawdown |
|---|---|---|---|---|
| MV baseline | 60.60% | 13.16% | 4.60 | -4.86% |
| MV tightened | 48.14% | 8.01% | 6.01 | -2.79% |
| S&P 500 (SPY) | 20.29% | 12.23% | 1.66 | -9.13% |
| Dow Jones (DIA) | 16.44% | 12.40% | 1.33 | -10.06% |

![results](e4_mv_risk_correction.png)
