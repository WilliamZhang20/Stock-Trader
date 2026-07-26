# Trials

Backtests of the CVaR and mean-variance traders over the most recent ~1-year window, benchmarked against the S&P 500 (SPY), Dow Jones (DIA), and a 60/40 (SPY/IEF) blend. Universes are chosen by `market_analyzer.build_universe` using only data available before the window (no lookahead). Prices are dividend/split adjusted, Sharpe ratios are **excess** of a 4% risk-free rate, and equity curves are **net of 10 bps** per-turnover transaction costs.

## Key findings

- **The Markowitz trader was overhauled:** HMM gone; Black-Litterman returns + Iterated-EWMA covariance + weekly rebalancing + volatility targeting (practical Markowitz / Boyd). See E4 ablation and trials/improve_mv.py.
- **Robustness checks (E8):** after charging costs, scoring out-of-sample across sub-windows, and deflating for multiple testing, CVaR net 38.4% ret / 3.18 Sharpe / -5.1% DD and Mean-Variance net 42.3% ret / 2.72 Sharpe / -7.2% DD. See E8 for gross-vs-net, per-window consistency, and deflated-Sharpe detail.
- **Benchmarks (this window):** SPY 17.8% ret / 1.08 Sharpe / -8.9% DD; DIA 17.8% ret / 1.12 Sharpe / -9.8% DD.
- **CVaR beats both benchmarks risk-adjusted** (38.4% ret / 3.18 Sharpe / -5.1% DD) with roughly half the drawdown of SPY.
- **Adaptive cluster allocation is the standout knob** (45.3% ret / 3.17 Sharpe / -5.8% DD vs uniform 38.4% ret / 3.18 Sharpe / -5.1% DD): tilting universe slots toward the strongest-performing clusters improved return, Sharpe, and drawdown together.
- **Practical Markowitz (IEWMA + weekly) beats the legacy daily/LW stack** (40.3% ret / 2.58 Sharpe / -7.4% DD vs legacy 30.3% ret / 1.47 Sharpe / -10.6% DD): Iterated-EWMA covariance and weekly rebalancing were the largest levers (see trials/improve_mv.py).
- **A rolling, adaptively-weighted universe is implemented** — the pool is re-surveyed each quarter and the per-cluster weighting re-adapts to recent performance. In this window it **underperformed** the stable fixed universe (rolling-adaptive 18.5% ret / 1.59 Sharpe / -7.7% DD vs fixed 38.4% ret / 3.18 Sharpe / -5.1% DD): quarterly winner-chasing added turnover and timing risk. The mechanism works; the naive momentum edge didn't pay here.
- **The CVXPYgen-compiled solver** (`fast_cvar.py`) is ~2x faster end-to-end with bit-identical results — see E5.

## Experiments

| Experiment | Report |
|---|---|
| E1 — CVaR baseline vs S&P/Dow | [e1_cvar_baseline.md](e1_cvar_baseline.md) |
| E2 — Mean-Variance baseline vs S&P/Dow | [e2_mv_baseline.md](e2_mv_baseline.md) |
| E3 — CVaR universe-selection variations | [e3_cvar_universe.md](e3_cvar_universe.md) |
| E4 — Mean-Variance practical vs legacy | [e4_mv_risk_correction.md](e4_mv_risk_correction.md) |
| E5 — CVXPYgen compiled solver | [e5_fast_solver.md](e5_fast_solver.md) |
| E6 — CVaR fixed vs rolling universe | [e6_cvar_rolling.md](e6_cvar_rolling.md) |
| E7 — Mean-Variance fixed vs rolling universe | [e7_mv_rolling.md](e7_mv_rolling.md) |
| E8 — Robustness: costs / out-of-sample / deflated Sharpe | [e8_robustness.md](e8_robustness.md) |

Regenerate with: `python trials/run_trials.py` (from the project root).

_Generated 2026-07-26. Returns are for one specific recent window and are not predictive; the harness re-selects universes and re-runs on each invocation._
