# E5 — CVXPYgen Compiled Solver vs Plain CVXPY

**Window:** 2025-06-10 → 2026-06-10 (252 trading days)  
**Universe selected as of:** 2023-12-24 (no lookahead)

Validates the CVXPYgen-compiled solver (`fast_cvar.py`): the same enhanced-CVaR problem, recast as a DPP-parametric program and compiled to C.

- Plain CVXPY backtest: **3.5s**
- Compiled backtest: **1.8s**  (**2.0x** faster)
- Max equity-curve difference: **1.43e-10** (identical within tolerance)

**Universe:** `['GLD', 'AVGO', 'UNH', 'HYG', 'JPM', 'XOM', 'MCD', 'GOOGL']`

## Results

| Strategy | Ann. Return | Ann. Vol | Sharpe | Max Drawdown |
|---|---|---|---|---|
| CVaR plain | 22.68% | 9.62% | 2.36 | -5.38% |
| CVaR compiled | 22.68% | 9.62% | 2.36 | -5.38% |
| S&P 500 (SPY) | 20.29% | 12.23% | 1.66 | -9.13% |
| Dow Jones (DIA) | 16.44% | 12.40% | 1.33 | -10.06% |

![results](e5_fast_solver.png)
