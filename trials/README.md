# Trials

Backtests of the CVaR and mean-variance traders over the most recent ~1-year window, benchmarked against the S&P 500 (SPY) and Dow Jones (DIA). Universes are chosen by `market_analyzer.build_universe` using only data available before the window (no lookahead).

## Key findings

- **Benchmarks (this window):** SPY 20.3% ret / 1.66 Sharpe / -9.1% DD; DIA 16.4% ret / 1.33 Sharpe / -10.1% DD.
- **CVaR beats both benchmarks risk-adjusted** (22.7% ret / 2.36 Sharpe / -5.4% DD) with roughly half the drawdown of SPY.
- **Adaptive cluster allocation is the standout knob** (34.1% ret / 2.77 Sharpe / -4.6% DD vs uniform 22.7% ret / 2.36 Sharpe / -5.4% DD): tilting universe slots toward the strongest-performing clusters improved return, Sharpe, and drawdown together.
- **Mean-variance is higher-octane but riskier**; the simple risk-tightening correction (tighter weight cap + higher risk aversion) cut volatility and drawdown while keeping strong returns (48.1% ret / 6.01 Sharpe / -2.8% DD vs baseline 60.6% ret / 4.60 Sharpe / -4.9% DD).
- **A rolling, adaptively-weighted universe is implemented** — the pool is re-surveyed each quarter and the per-cluster weighting re-adapts to recent performance. In this window it **underperformed** the stable fixed universe (rolling-adaptive 8.8% ret / 0.69 Sharpe / -12.2% DD vs fixed 22.7% ret / 2.41 Sharpe / -5.8% DD): quarterly winner-chasing added turnover and timing risk. The mechanism works; the naive momentum edge didn't pay here.
- **The CVXPYgen-compiled solver** (`fast_cvar.py`) is ~2x faster end-to-end with bit-identical results — see E5.

## Experiments

| Experiment | Report |
|---|---|
| E1 — CVaR baseline vs S&P/Dow | [e1_cvar_baseline.md](e1_cvar_baseline.md) |
| E2 — Mean-Variance baseline vs S&P/Dow | [e2_mv_baseline.md](e2_mv_baseline.md) |
| E3 — CVaR universe-selection variations | [e3_cvar_universe.md](e3_cvar_universe.md) |
| E4 — Mean-Variance risk-tightening | [e4_mv_risk_correction.md](e4_mv_risk_correction.md) |
| E5 — CVXPYgen compiled solver | [e5_fast_solver.md](e5_fast_solver.md) |
| E6 — CVaR fixed vs rolling universe | [e6_cvar_rolling.md](e6_cvar_rolling.md) |
| E7 — Mean-Variance fixed vs rolling universe | [e7_mv_rolling.md](e7_mv_rolling.md) |

Regenerate with: `python trials/run_trials.py` (from the project root).

_Generated 2026-06-11. Returns are for one specific recent window and are not predictive; the harness re-selects universes and re-runs on each invocation._
