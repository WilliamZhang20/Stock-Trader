# Stock Trader

This repository is used for a trading bot that applies convex optimization to make decisions, via the [CVXPY](https://www.cvxpy.org/) library developed at [Stanford](https://github.com/cvxgrp).

As of now, it trades on fake money using an Alpaca Paper account that started with 100K dollars.

It also only has working stock/ETF trading, but not options trading (that will be ready soon).

## Stock Trader Versions

The `prev_versions` folder contains previous end-to-end implementations of securities portfolio allocations in CVXPY.

Now the main directory has only two primary trader files, which highlight two different strategies: mean-variance optimization and conditional value-at-risk optimization.

An inspiration for my exploration of other strategies was [this](https://developer.nvidia.com/blog/accelerating-real-time-financial-decisions-with-quantitative-portfolio-optimization/) article by NVIDIA.

The files which implement those strategies are easily named `mean_variance_trader.py` and `cvar_trader.py`.

The basic mathematical formulation is given below, while a more deep dive is given in the "Trading Strategy" section below.

Both strategies solve for $w$, a vector where value $i$ is the portion of portfolio allocated to asset $i$ in a set array of market ticker symbols. This array is also known as the "universe" of assets.

The **mean-variance** optimizer solves for $w$ in the problem below:

```math
\begin{aligned}
\max_{\mathbf{w}} \quad & \mathbf{\mu}^\top \mathbf{w} - \frac{\lambda}{2} \mathbf{w}^\top \mathbf{\Sigma} \mathbf{w} \\
\text{s.t.} \quad & \mathbf{1}^\top \mathbf{w} = \text{MAX\_INVEST}
\end{aligned}
```

using a quadratic program in the `OSQP` solver from CVXPY. $\Sigma$ is the Ledoit-Wolf shrinkage covariance matrix of the universe, and $\mu$ is the expected-return estimate.

Rather than a raw sample mean (a notoriously noisy input that makes textbook Markowitz overfit), $\mu$ is a **Black-Litterman posterior**: momentum/EWMA "views" are shrunk toward a market-equilibrium prior obtained by reverse optimization. Portfolio risk is then stabilized by an explicit **volatility target** that scales gross exposure toward a fixed annualized vol. This "practical Markowitz" design replaces an earlier, fragile HMM regime-switching risk model that has since been removed. See [trials/e8_robustness.md](trials/e8_robustness.md) for net-of-cost, out-of-sample, benchmarked results.

The **CVaR** optimizer solves:

```math
\begin{aligned}\max_{\mathbf{w}, z, \mathbf{u}} \quad & \lambda_\text{ret} \sum_{i=1}^{N} \bar{R}_i w_i - \lambda_\text{cvar} \left[ z + \frac{1}{(1-\alpha) T} \sum_{t=1}^{T} u_t \right] \\
\text{s.t.} \quad &
\mathbf{1}^\top \mathbf{w} = w_{\text{total}} \le \text{MAX\_INVEST}, \\
& 0 \le w_i \le \text{MAX\_WEIGHT}, \quad i = 1,\dots,N, \\
& u_t \ge - \mathbf{R}_t^\top \mathbf{w} - z, \quad t = 1,\dots,T, \\
& u_t \ge 0, \quad t = 1,\dots,T, \\
& z \ge 0.
\end{aligned}
```

using a linear program, where:

$z$ = the threshold beyond which losses are considered “bad” (VaR).

$u_t$ = how much worse the portfolio did than that threshold on day $t$

$T$ = the number of historical return samples used to compute the CVaR.

$\mathbf{w}$ = portfolio weights vector.

$\lambda_\text{ret}$ = coefficient for the expected return term.

$\lambda_\text{cvar}$ = coefficient for the CVaR term.

$\bar{R}_i$ = expected return of asset $i$.

$\alpha$ = confidence level for CVaR (e.g., 0.95).

$\mathbf{R}_t$ = vector of asset returns at time $t$.

$N$ = number of assets in the universe.

The CVaR strategy has historically shown more robust, lower-drawdown performance than mean-variance. For current, apples-to-apples numbers (net of costs, benchmarked, and checked out-of-sample) see the [trials](trials/README.md) folder rather than any single headline figure quoted here.

## Robustness & Caveats

A high backtested annualized return (e.g. the ~43% seen on some mean-variance runs) should invite skepticism. Here is how each of the four usual objections is handled in this repo; the [E8 robustness report](trials/e8_robustness.md) quantifies all of them.

- **Transaction costs.** Backtests charge `COST_BPS` (default 10 bps) on L1 turnover at every rebalance, deducted directly from the equity curve in both `cvar_trader.py` and `mean_variance_trader.py`. E8 reports gross vs net side by side so the cost drag is explicit. (Previously, costs were ignored and turnover was only a soft penalty in the objective.)
- **Overfitting.** The strategies carry many hand-tuned constants, and several configurations were tried on the same data. E8 (a) scores each strategy on consecutive, non-overlapping out-of-sample sub-windows (consistency, not one lucky year) and (b) reports a **Deflated Sharpe Ratio** that discounts for having tried ~N configurations. The Markowitz overhaul also *reduces* the number of free knobs: volatility targeting and Black-Litterman replace the HMM's opaque state-to-multiplier machinery with interpretable parameters.
- **Data leakage.** Universe selection is point-in-time (`build_universe(end_date=...)`), and every trade uses only returns strictly *before* the trade day (now enforced with `assert`s in both backtests; a look-ahead bug in the mean-variance window was fixed). Prices are split/dividend adjusted. The remaining, honestly-disclosed bias is **survivorship**: `CANDIDATE_POOL` lists tickers still trading today, so absolute returns are optimistic (documented in `market_analyzer.py`).
- **Benchmark selection.** Benchmarks now use dividend-adjusted **total-return** prices, Sharpe ratios are computed **excess** of a ~4% risk-free rate (not the old `rf=0`), and a **60/40 (SPY/IEF)** comparator is included alongside SPY/DIA because these strategies often hold cash and run below-market volatility.

## Getting Started

Alpaca SDK API keys are stored as environment variables and fetched using `os.environ` for safety. 

Besides stock trading strategy explorations outlined above in all the `cvx_trader_v(x).py`, I have begun experiments with Black-Scholes options trading on the Alpaca inside `black_scholes_options.py`.

The script `plot_risk_return.py` assists in gauging the annualized risk-return trade-off between various assets. When plotting, one can observe that higher returns often lead to higher risk.

The Python script `purge_helper.py` is for helping to get rid of any shares with fractional value on Alpaca.

## Trading Strategy

Two strategies were explored/researched/experimented with: Mean-Variance (Markowitz) portfolio optimization, and Conditional Value-at-Risk (CVaR) optimization.

### Mean-Variance Optimization

The strategy used in the trading algorithm is mean-variance optimization. In any investment, we want to maximize gain with the least amount of risk within a certain trading period.

Judging an asset's ability to increase as well as its risk can be based on historical data, as well as current signals.

For example, the picture below is an annualized risk-return plot for a large number of assets from January 2024 to September 2025. Notice that higher gain tends to come with more risk, although there are some exceptions.

From the various market signals, the algorithm simply determines portfolio allocation to various assets. If we set the parameters to be more strongly risk-averse, then higher proportions will be allocated to lower-risk assets with maximum possible returns. Similarly, when we are more risk-tolerant, the algorithm will be willing to allocate more to assets with more risk, while seeking the best ROI.

In essence, the optimization is somewhat multi-objective.

<img width="800" height="600" alt="risk_return_plot" src="https://github.com/user-attachments/assets/e99e19cc-4f17-40a7-aa24-2ce13e8bff87" />

### Conditional Value-at-Risk

While the previous mean-variance optimization portfolio managed risk via covariance, it treated all deviations from the mean as "bad", whereas proper portfolios only penalize downside risk.

Conditional Value-at-Risk (CVaR), also called Expected Shortfall, is a statistical measure that focuses only on the tail of the loss distribution.

For a given confidence level $\alpha$ (e.g., 95%), the Value-at-Risk (VaR) is the loss threshold such that with probability $\alpha$, your losses do not exceed it.

```math
\mathrm{VaR}_{\alpha}(L) = \inf\left\{ \ell : \mathbb{P}(L \le \ell) \ge \alpha \right\}
```

VaR tells you a threshold, but not the expected loss beyond that threshold.

CVaR is the average loss in the worst $1 - \alpha$ fraction of cases:
```math
\mathrm{CVaR}_{\alpha}(L) = \mathbb{E}\!\left[ L \mid L \ge \mathrm{VaR}_{\alpha}(L) \right]
```

- Intuition: "If things go badly beyond the 95% worst-case threshold, the expected loss is CVaR."


[Rockafellar and Uryasev](https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf) showed that CVaR can be expressed as a convex optimization problem, in fact as a linear program, making it easy to solve and computationally tractable.

## Accelerated Solver (CVXPYgen)

The CVaR optimizer can optionally run a [CVXPYgen](https://github.com/cvxgrp/cvxpygen)-generated C solver (`fast_cvar.py`) instead of the pure-Python CVXPY path. It produces bit-identical weights at roughly 2x faster end-to-end (~6x per solve), which matters across a long backtest where the optimizer is called on every rebalance.

To use it, just add the `--fast` flag:

```bash
python cvar_trader.py --backtest --fast
```

The first run **generates and compiles** the C extension for the current problem size (number of assets `N` and return-window length `T`), which takes ~25 s. Compiled solvers are cached on disk as `enhanced_cvar_code_<N>_<T>/`, so every subsequent run with the same dimensions skips compilation and is fast immediately. If the solver fails to build or load for any reason, the code automatically falls back to the standard CVXPY path — `--fast` never breaks a run.

**Requirements:** a C compiler must be on `PATH` (this repo was built with mingw64 gcc; MSVC also works). On machines with a global `user = true` pip config, set `PIP_USER=0` before running so CVXPYgen's `pip --target` build step doesn't conflict (`fast_cvar.py` already sets this around code generation, but exporting it yourself avoids edge cases):

```bash
# PowerShell
$env:PIP_USER = "0"; python cvar_trader.py --backtest --fast
```

The generated `enhanced_cvar_code_*/` directories are build artifacts and are git-ignored; delete them anytime to force a clean re-compile.
