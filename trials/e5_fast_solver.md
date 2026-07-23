# E5 — CVXPYgen Compiled Solver vs Plain CVXPY

**Window:** 2025-07-22 → 2026-07-22 (252 trading days)  
**Universe selected as of:** 2022-06-14 (no lookahead)

Validates the CVXPYgen-compiled solver (`fast_cvar.py`): the same enhanced-CVaR problem, recast as a DPP-parametric program and compiled to C.

- Plain CVXPY backtest: **15.7s**
- Compiled backtest: **6.4s**  (**2.5x** faster)
- Max equity-curve difference: **3.88e-11** (identical within tolerance)

**Universe:** `['KO', 'TSLA', 'CAT', 'EEM', 'LQD', 'SLV', 'PDBC', 'UNH']`

## Results

| Strategy | Ann. Return | Ann. Vol | Sharpe (excess) | Max Drawdown |
|---|---|---|---|---|
| CVaR plain | 38.90% | 10.81% | 3.23 | -5.06% |
| CVaR compiled | 38.90% | 10.81% | 3.23 | -5.06% |
| S&P 500 (SPY) | 20.17% | 12.66% | 1.28 | -8.88% |
| Dow Jones (DIA) | 18.96% | 12.28% | 1.22 | -9.76% |
| 60/40 (SPY/IEF) | 12.68% | 8.20% | 1.06 | -6.00% |

![results](e5_fast_solver.png)
