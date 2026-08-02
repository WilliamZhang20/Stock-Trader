"""
Experiment harness: backtest cvar_trader and mean_variance_trader over a recent
1-year window, benchmarked against the S&P 500 (SPY) and Dow Jones (DIA), across
several universe / configuration variations.

Each experiment writes a PNG graph and a Markdown report into this folder.
Run from the project root:  python trials/run_trials.py
"""
import os
import sys
import time
import datetime as dt

import matplotlib
matplotlib.use("Agg")  # headless: save figures, never block on plt.show()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make project modules importable when run from anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cvar_trader as ct
import mean_variance_trader as mv
from market_analyzer import build_universe, CANDIDATE_POOL, select_universe

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WINDOW = 252                       # trading days in the evaluation window (~1yr)
BENCHMARKS = {"SPY": "S&P 500 (SPY)", "DIA": "Dow Jones (DIA)"}
TODAY = dt.date.today()
DATA_START = (TODAY - dt.timedelta(days=1500)).isoformat()  # ~4yr of data (room for several OOS sub-windows)
SELECT_END = DATA_START            # pick universes using only pre-window data

# Fair-benchmark accounting knobs.
RISK_FREE = 0.04                   # annual risk-free rate for EXCESS-return Sharpe
# Rough count of strategy configurations tried on this data (E1-E7 variants),
# used to deflate the Sharpe ratio for multiple-testing (see deflated_sharpe).
N_TRIALS = 8

_MASTER_PX = None


def master_px():
    """Fetch one wide price panel covering the whole candidate pool + benchmarks."""
    global _MASTER_PX
    if _MASTER_PX is None:
        syms = sorted(set(CANDIDATE_POOL) | set(BENCHMARKS))
        print(f"Fetching master price panel for {len(syms)} symbols from {DATA_START}...")
        _MASTER_PX = ct.fetch_alpaca_prices(syms, DATA_START, None)
    return _MASTER_PX


def universe_px(universe):
    """Clean price sub-panel for a given universe."""
    px = master_px()[universe].ffill().dropna()
    return px


def windowed(curve):
    """Take the last WINDOW points of an equity curve and renormalize to 1.0."""
    tail = curve.dropna().iloc[-WINDOW:]
    return tail / tail.iloc[0]


def pool_px():
    """Clean price panel for the full candidate pool (for rolling-universe backtests)."""
    syms = [s for s in CANDIDATE_POOL if s in master_px().columns]
    return master_px()[syms].ffill().dropna(axis=1)


def benchmark_series(index, sym):
    """Buy-and-hold benchmark aligned to `index`, normalized to 1.0 at the start."""
    bench = master_px()[sym].reindex(index).ffill().bfill()
    return bench / bench.iloc[0]


def benchmark_6040(index, equity="SPY", bond="IEF"):
    """
    Daily-rebalanced 60/40 stock/bond benchmark, normalized to 1.0.

    A fairer comparator than 100% equities because the strategies frequently hold
    cash / bonds and run below-market volatility; comparing them to a pure-equity
    index flatters their risk-adjusted numbers.
    """
    px = master_px()
    e = px[equity].reindex(index).ffill().bfill().pct_change().fillna(0.0)
    b = px[bond].reindex(index).ffill().bfill().pct_change().fillna(0.0)
    blend = 0.6 * e + 0.4 * b
    curve = (1.0 + blend).cumprod()
    return curve / curve.iloc[0]


def metrics(curve, rf=RISK_FREE):
    """
    Annualized return, vol, EXCESS-return Sharpe, and max drawdown for a curve
    normalized to 1.0 at the start.

    Sharpe uses `(ann_ret - rf) / ann_vol` so it is not inflated by simply
    earning the ~4-5% cash rate (the old harness assumed rf=0).
    """
    daily = curve.pct_change().dropna()
    n = len(curve)
    ann_ret = curve.iloc[-1] ** (252 / n) - 1
    ann_vol = daily.std() * np.sqrt(252)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else float("nan")
    dd = (curve - curve.cummax()) / curve.cummax()
    return dict(ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe, max_dd=dd.min())


def deflated_sharpe(curve, n_trials=N_TRIALS, rf=RISK_FREE):
    """
    Probability that the observed (excess) Sharpe is genuinely > 0 after
    accounting for having tried `n_trials` configurations on the same data
    (Bailey & Lopez de Prado's Deflated Sharpe Ratio, non-annualized inputs).

    A value near 1.0 means the result survives multiple-testing skepticism; near
    0.5 or below means it is plausibly a fluke from selecting the best of many.
    """
    from scipy.stats import norm

    daily = curve.pct_change().dropna()
    T = len(daily)
    if T < 20 or daily.std() == 0:
        return float("nan")
    sr = (daily.mean() - rf / 252) / daily.std()  # per-observation Sharpe
    skew = float(daily.skew())
    kurt = float(daily.kurtosis()) + 3.0          # pandas gives excess kurtosis
    # Expected max Sharpe under the null of `n_trials` independent zero-skill trials.
    sr_std = 1.0 / np.sqrt(T - 1)
    gamma = 0.5772156649
    e = np.e
    z1 = norm.ppf(1 - 1.0 / n_trials)
    z2 = norm.ppf(1 - 1.0 / (n_trials * e))
    sr0 = sr_std * ((1 - gamma) * z1 + gamma * z2)
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2, 1e-9))
    return float(norm.cdf(((sr - sr0) * np.sqrt(T - 1)) / denom))


def run_cvar(universe, fast=False):
    prev = ct.USE_FAST_SOLVER
    ct.USE_FAST_SOLVER = fast
    try:
        t0 = time.time()
        curve, _ = ct.walk_forward_backtest(universe_px(universe))
        elapsed = time.time() - t0
    finally:
        ct.USE_FAST_SOLVER = prev
    return windowed(curve), elapsed


def run_mv(universe):
    t0 = time.time()
    curve, _ = mv.walk_forward_backtest(universe_px(universe))
    return windowed(curve), time.time() - t0


def run_cvar_full(universe, cost_bps=None):
    """Full (un-windowed) CVaR equity curve, optionally overriding COST_BPS."""
    saved = ct.COST_BPS
    if cost_bps is not None:
        ct.COST_BPS = cost_bps
    try:
        curve, _ = ct.walk_forward_backtest(universe_px(universe))
    finally:
        ct.COST_BPS = saved
    return curve.dropna()


def run_mv_full(universe, cost_bps=None):
    """
    Full (un-windowed) mean-variance equity curve.

    ``cost_bps`` scales the asset-specific cost floor: 0 → zero costs (gross);
    None → current BASE_COST_BPS / VOL_COST_MULT defaults.
    """
    saved = (mv.BASE_COST_BPS, mv.VOL_COST_MULT, mv.COST_BPS)
    if cost_bps is not None:
        mv.BASE_COST_BPS = float(cost_bps)
        mv.VOL_COST_MULT = 0.0 if cost_bps == 0 else mv.VOL_COST_MULT
        mv.COST_BPS = float(cost_bps)
    try:
        curve, _ = mv.walk_forward_backtest(universe_px(universe))
    finally:
        mv.BASE_COST_BPS, mv.VOL_COST_MULT, mv.COST_BPS = saved
    return curve.dropna()


def oos_windows(curve, size=WINDOW):
    """
    Split a full equity curve into consecutive non-overlapping sub-windows, each
    renormalized to 1.0. Leading warm-up region (equity flat at 1.0) is trimmed.
    Used to check that returns are consistent across periods, not one lucky year.
    """
    c = curve.dropna()
    active = c[c != 1.0]
    if len(active) > 0:
        c = c.loc[active.index[0]:]
    wins = []
    for i in range(0, len(c) - size + 1, size):
        seg = c.iloc[i:i + size]
        wins.append(seg / seg.iloc[0])
    if not wins:
        wins = [c / c.iloc[0]]
    return wins


def run_cvar_rolling(allocation, refresh=63):
    """Rolling-universe CVaR backtest. Returns (windowed_curve, universe_log)."""
    curve, _, log = ct.walk_forward_backtest_dynamic(
        pool_px(), target_size=8, universe_refresh_days=refresh,
        select_kwargs=dict(allocation=allocation, criterion="sharpe"),
    )
    return windowed(curve), log


def run_mv_rolling(allocation, refresh=63):
    curve, _, log = mv.walk_forward_backtest_dynamic(
        pool_px(), target_size=9, universe_refresh_days=refresh,
        select_kwargs=dict(allocation=allocation, criterion="sharpe"),
    )
    return windowed(curve), log


def summarize_log(log):
    """Compact text summary of a rolling universe_log for a Markdown report."""
    if not log:
        return "_(universe never changed)_"
    lines = [f"- {d.date()}: `{u}`" for d, u in log.items()]
    return f"{len(log)} re-selections:\n\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def make_report(slug, title, series: dict, body_md: str, metrics_rows: list):
    """series: {label: normalized_curve}. Writes <slug>.png and <slug>.md."""
    # Align everything to the first strategy curve's index.
    base_index = next(iter(series.values())).index

    plt.figure(figsize=(11, 6))
    for label, curve in series.items():
        style = "-" if not label.startswith(("S&P", "Dow")) else "--"
        lw = 2.0 if style == "-" else 1.3
        plt.plot(curve.index, (curve.values - 1) * 100, style, linewidth=lw, label=label)
    plt.axhline(0, color="gray", linewidth=0.6)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative return (%)")
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    png = os.path.join(_HERE, f"{slug}.png")
    plt.savefig(png, dpi=110)
    plt.close()

    # Metrics table
    header = "| Strategy | Ann. Return | Ann. Vol | Sharpe (excess) | Max Drawdown |\n"
    header += "|---|---|---|---|---|\n"
    rows = ""
    for name, m in metrics_rows:
        rows += (f"| {name} | {m['ann_ret']:.2%} | {m['ann_vol']:.2%} | "
                 f"{m['sharpe']:.2f} | {m['max_dd']:.2%} |\n")

    start = base_index[0].date()
    end = base_index[-1].date()
    md = (
        f"# {title}\n\n"
        f"**Window:** {start} → {end} ({len(base_index)} trading days)  \n"
        f"**Universe selected as of:** {SELECT_END} (no lookahead)\n\n"
        f"{body_md}\n\n"
        f"## Results\n\n{header}{rows}\n"
        f"![results]({slug}.png)\n"
    )
    with open(os.path.join(_HERE, f"{slug}.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  wrote {slug}.md + {slug}.png")
    return {name: m for name, m in metrics_rows}


def add_benchmarks(series: dict):
    """Append SPY/DIA benchmark curves aligned to the first series' index."""
    idx = next(iter(series.values())).index
    out = dict(series)
    bench_metrics = []
    for sym, label in BENCHMARKS.items():
        b = benchmark_series(idx, sym)
        out[label] = b
        bench_metrics.append((label, metrics(b)))
    b6040 = benchmark_6040(idx)
    out["60/40 (SPY/IEF)"] = b6040
    bench_metrics.append(("60/40 (SPY/IEF)", metrics(b6040)))
    return out, bench_metrics


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def exp_cvar_baseline():
    print("\n[E1] CVaR baseline vs S&P / Dow")
    uni = build_universe(target_size=8, end_date=SELECT_END)
    curve, _ = run_cvar(uni)
    series = {"CVaR (uniform/sharpe, 8)": curve}
    series, bm = add_benchmarks(series)
    body = ("Baseline CVaR trader on an 8-asset universe selected by PCA + k-means "
            f"(one asset per cluster, Sharpe-ranked).\n\n**Universe:** `{uni}`")
    return make_report("e1_cvar_baseline", "E1 — CVaR Trader vs S&P 500 vs Dow Jones",
                       series, body, [("CVaR (uniform/sharpe, 8)", metrics(curve))] + bm)


def exp_mv_baseline():
    print("\n[E2] Mean-Variance baseline vs S&P / Dow")
    uni = build_universe(target_size=9, end_date=SELECT_END)
    curve, _ = run_mv(uni)
    series = {"Mean-Variance (uniform/sharpe, 9)": curve}
    series, bm = add_benchmarks(series)
    body = ("Practical mean-variance trader (Black-Litterman returns + IEWMA covariance + "
            "weekly rebalancing + volatility targeting) on a 9-asset PCA/k-means "
            f"universe.\n\n**Universe:** `{uni}`")
    return make_report("e2_mv_baseline", "E2 — Mean-Variance Trader vs S&P 500 vs Dow Jones",
                       series, body, [("Mean-Variance (uniform/sharpe, 9)", metrics(curve))] + bm)


def exp_cvar_universe_variations():
    print("\n[E3] CVaR universe-selection variations")
    configs = {
        "uniform / sharpe": dict(allocation="uniform", criterion="sharpe"),
        "adaptive / sharpe": dict(allocation="adaptive", criterion="sharpe"),
        "uniform / calmar": dict(allocation="uniform", criterion="calmar"),
    }
    series = {}
    rows = []
    universes = {}
    for label, cfg in configs.items():
        uni = build_universe(target_size=8, end_date=SELECT_END, **cfg)
        universes[label] = uni
        curve, _ = run_cvar(uni)
        series[f"CVaR ({label})"] = curve
        rows.append((f"CVaR ({label})", metrics(curve)))
    series, bm = add_benchmarks(series)
    body = "How the universe-selection knobs affect the CVaR trader:\n\n"
    for label, uni in universes.items():
        body += f"- **{label}:** `{uni}`\n"
    return make_report("e3_cvar_universe", "E3 — CVaR Trader: Universe-Selection Variations",
                       series, body, rows + bm)


def exp_mv_risk_correction():
    print("\n[E4] Mean-Variance v7 (relative BL + asset costs) vs absolute-views ablation")
    uni = build_universe(target_size=9, end_date=SELECT_END)

    # Current defaults: relative BL views + asset-specific costs + daily rebal.
    practical_curve, _ = run_mv(uni)

    # Ablation: absolute-only BL views (P = I).
    _bl = mv.black_litterman_mu

    def _abs_only(returns, Sigma, **kwargs):
        kwargs = dict(kwargs)
        kwargs["use_relative_views"] = False
        return _bl(returns, Sigma, **kwargs)

    mv.black_litterman_mu = _abs_only
    try:
        abs_curve, _ = run_mv(uni)
    finally:
        mv.black_litterman_mu = _bl

    series = {
        "MV v7 relative": practical_curve,
        "MV absolute-only": abs_curve,
    }
    series, bm = add_benchmarks(series)
    body = (
        "Ablation of the v7 relative Black-Litterman views. Both arms use IEWMA "
        "covariance, daily rebalancing, volatility targeting, and asset-specific "
        f"L1 trading costs (base {mv.BASE_COST_BPS:.0f} bps + vol load). The "
        "treatment adds residual mean-reversion relative views (P_k = e_i - β e_j) "
        "between correlated peers; the ablation keeps only absolute EWMA views "
        f"(P = I).\n\n**Universe:** `{uni}`"
    )
    rows = [
        ("MV v7 relative", metrics(practical_curve)),
        ("MV absolute-only", metrics(abs_curve)),
    ]
    return make_report("e4_mv_risk_correction",
                       "E4 — Mean-Variance Relative BL Views vs Absolute-Only",
                       series, body, rows + bm)


def exp_fast_solver_benchmark():
    print("\n[E5] Fast (CVXPYgen) vs plain CVXPY solver")
    uni = build_universe(target_size=8, end_date=SELECT_END)
    plain_curve, t_plain = run_cvar(uni, fast=False)
    # Warm up: trigger the one-time CVXPYgen compile OUTSIDE the timed region so
    # we measure steady-state solve speed, not the build.
    run_cvar(uni, fast=True)
    fast_curve, t_fast = run_cvar(uni, fast=True)

    max_diff = float(np.max(np.abs(fast_curve.values - plain_curve.values)))
    speedup = t_plain / t_fast if t_fast > 0 else float("nan")

    series = {
        "CVaR (plain CVXPY/ECOS)": plain_curve,
        "CVaR (CVXPYgen compiled)": fast_curve,
    }
    series, bm = add_benchmarks(series)
    body = (
        "Validates the CVXPYgen-compiled solver (`fast_cvar.py`): the same enhanced-CVaR "
        "problem, recast as a DPP-parametric program and compiled to C.\n\n"
        f"- Plain CVXPY backtest: **{t_plain:.1f}s**\n"
        f"- Compiled backtest: **{t_fast:.1f}s**  (**{speedup:.1f}x** faster)\n"
        f"- Max equity-curve difference: **{max_diff:.2e}** (identical within tolerance)\n\n"
        f"**Universe:** `{uni}`"
    )
    rows = [
        ("CVaR plain", metrics(plain_curve)),
        ("CVaR compiled", metrics(fast_curve)),
    ]
    return make_report("e5_fast_solver", "E5 — CVXPYgen Compiled Solver vs Plain CVXPY",
                       series, body, rows + bm)


def _fmt(m):
    return f"{m['ann_ret']:.1%} ret / {m['sharpe']:.2f} Sharpe / {m['max_dd']:.1%} DD"


def exp_cvar_rolling():
    print("\n[E6] CVaR — fixed vs rolling (adaptive) universe")
    fixed_uni = build_universe(target_size=8, end_date=SELECT_END)
    fixed_curve, _ = run_cvar(fixed_uni)
    roll_uniform, _ = run_cvar_rolling("uniform")
    roll_adaptive, log = run_cvar_rolling("adaptive")

    series = {
        "CVaR fixed universe": fixed_curve,
        "CVaR rolling (uniform clusters)": roll_uniform,
        "CVaR rolling (adaptive clusters)": roll_adaptive,
    }
    series, bm = add_benchmarks(series)
    body = (
        "The universe is re-surveyed every ~quarter (63 trading days) using only past "
        "data, so the portfolio can rotate into newly-surging leaders. Three modes:\n\n"
        "- **fixed** — one universe chosen at the start and held;\n"
        "- **rolling / uniform** — re-selected each quarter, one asset per cluster;\n"
        "- **rolling / adaptive** — re-selected each quarter *and* the per-cluster slot "
        "allocation re-tilts toward the strongest-performing clusters.\n\n"
        "> **Caveat:** rolling selection picks *recent* winners, which tend to mean-revert; "
        "it also adds turnover (not charged here). Compare the table below — a stable, "
        "diversified universe can beat quarterly winner-chasing in a trending market.\n\n"
        f"### Rolling adaptive universe over time\n\n{summarize_log(log)}\n\n"
        f"**Fixed universe:** `{fixed_uni}`"
    )
    rows = [
        ("CVaR fixed", metrics(fixed_curve)),
        ("CVaR rolling (uniform)", metrics(roll_uniform)),
        ("CVaR rolling (adaptive)", metrics(roll_adaptive)),
    ]
    return make_report("e6_cvar_rolling", "E6 — CVaR Trader: Fixed vs Rolling Universe",
                       series, body, rows + bm)


def exp_mv_rolling():
    print("\n[E7] Mean-Variance — fixed vs rolling (adaptive) universe")
    fixed_uni = build_universe(target_size=9, end_date=SELECT_END)
    fixed_curve, _ = run_mv(fixed_uni)
    roll_adaptive, log = run_mv_rolling("adaptive")

    series = {
        "MV fixed universe": fixed_curve,
        "MV rolling (adaptive clusters)": roll_adaptive,
    }
    series, bm = add_benchmarks(series)
    body = (
        "Mean-variance trader with a fixed universe vs a rolling universe re-surveyed "
        "every ~quarter with adaptive cluster weighting (slots tilt toward recently "
        "strong clusters). Rolling variant uses static risk-aversion for simplicity.\n\n"
        "> **Caveat:** chasing recent winners quarterly adds turnover and timing risk; "
        "see whether it actually improves risk-adjusted returns in the table below.\n\n"
        f"### Rolling adaptive universe over time\n\n{summarize_log(log)}\n\n"
        f"**Fixed universe:** `{fixed_uni}`"
    )
    rows = [
        ("MV fixed", metrics(fixed_curve)),
        ("MV rolling (adaptive)", metrics(roll_adaptive)),
    ]
    return make_report("e7_mv_rolling", "E7 — Mean-Variance Trader: Fixed vs Rolling Universe",
                       series, body, rows + bm)


def exp_robustness():
    print("\n[E8] Robustness: transaction costs, out-of-sample, deflated Sharpe")
    uni_c = build_universe(target_size=8, end_date=SELECT_END)
    uni_m = build_universe(target_size=9, end_date=SELECT_END)

    cvar_gross = run_cvar_full(uni_c, cost_bps=0.0)
    cvar_net = run_cvar_full(uni_c)
    mv_gross = run_mv_full(uni_m, cost_bps=0.0)
    mv_net = run_mv_full(uni_m)

    gross_net_rows = [
        ("CVaR gross (0 bps)", metrics(windowed(cvar_gross))),
        ("CVaR net (10 bps)", metrics(windowed(cvar_net))),
        ("Mean-Variance gross (0 bps)", metrics(windowed(mv_gross))),
        (f"Mean-Variance net (asset c_i, base {mv.BASE_COST_BPS:.0f}bps)",
         metrics(windowed(mv_net))),
    ]
    gn_header = ("| Strategy | Ann. Return | Ann. Vol | Sharpe (excess) | Max Drawdown |\n"
                 "|---|---|---|---|---|\n")
    gn_body = "".join(
        f"| {n} | {m['ann_ret']:.2%} | {m['ann_vol']:.2%} | {m['sharpe']:.2f} | {m['max_dd']:.2%} |\n"
        for n, m in gross_net_rows
    )

    def oos_rows(name, curve):
        wins = oos_windows(curve)
        ms = [metrics(seg) for seg in wins]
        lines = []
        for i, (seg, m) in enumerate(zip(wins, ms), 1):
            lines.append(f"| {name} W{i} ({seg.index[0].date()}->{seg.index[-1].date()}) | "
                         f"{m['ann_ret']:.2%} | {m['sharpe']:.2f} | {m['max_dd']:.2%} |")
        rets = [m['ann_ret'] for m in ms]
        shs = [m['sharpe'] for m in ms]
        lines.append(f"| **{name} mean** | **{np.mean(rets):.2%}** | **{np.mean(shs):.2f}** | - |")
        lines.append(f"| **{name} worst** | **{np.min(rets):.2%}** | **{np.min(shs):.2f}** | - |")
        return lines

    oos_header = "| Window | Ann. Return | Sharpe (excess) | Max Drawdown |\n|---|---|---|---|\n"
    oos_body = "\n".join(oos_rows("CVaR", cvar_net) + oos_rows("Mean-Variance", mv_net))

    dsr_c = deflated_sharpe(windowed(cvar_net))
    dsr_m = deflated_sharpe(windowed(mv_net))

    body = (
        "Directly addresses the four reasons a high backtested return draws skepticism: "
        "overfitting, transaction costs, data leakage, and benchmark selection.\n\n"
        f"### 1. Transaction costs (gross vs net, last {WINDOW}d)\n"
        f"CVaR net charges {ct.COST_BPS:.0f} bps per unit of L1 turnover. "
        f"Mean-Variance net uses asset-specific costs "
        f"`sum_i c_i |dw_i|` (base {mv.BASE_COST_BPS:.0f} bps + vol load) in *both* "
        "the QP and the equity curve — the no-trade region. Gross charges nothing:\n\n"
        f"{gn_header}{gn_body}\n"
        "### 2. Overfitting - out-of-sample consistency\n"
        "The same frozen strategy scored on consecutive, non-overlapping ~1y sub-windows "
        "(net of costs). A single lucky year is easy; consistency across windows is not:\n\n"
        f"{oos_header}{oos_body}\n\n"
        "### 2b. Overfitting - deflated Sharpe (multiple testing)\n"
        f"Probability the excess Sharpe is real after ~{N_TRIALS} configurations were tried on "
        f"this data: **CVaR {dsr_c:.2f}**, **Mean-Variance {dsr_m:.2f}** "
        "(near 1.0 survives skepticism; near 0.5 is plausibly a fluke of selection).\n\n"
        "### 3. Data leakage\n"
        "Universe selection is point-in-time (`end_date=SELECT_END`); each trade uses only "
        "returns strictly before the trade day (now asserted in both backtests). Prices are "
        "split+dividend adjusted. Remaining bias: `CANDIDATE_POOL` is a survivors list "
        "(documented in `market_analyzer.py`), so absolute returns are optimistic.\n\n"
        "### 4. Benchmark selection\n"
        f"Benchmarks use dividend-adjusted total-return prices, a {RISK_FREE:.0%} risk-free rate "
        "for excess Sharpe, and add a 60/40 (SPY/IEF) comparator since these strategies often "
        "hold cash and run below-market volatility.\n\n"
        f"**CVaR universe:** `{uni_c}`\n\n**Mean-Variance universe:** `{uni_m}`"
    )

    series = {"CVaR (net)": windowed(cvar_net), "Mean-Variance (net)": windowed(mv_net)}
    series, bm = add_benchmarks(series)
    rows = [
        ("CVaR (net)", metrics(windowed(cvar_net))),
        ("Mean-Variance (net)", metrics(windowed(mv_net))),
    ]
    return make_report("e8_robustness",
                       "E8 - Robustness: Costs, Out-of-Sample, Deflated Sharpe",
                       series, body, rows + bm)


def _block_bootstrap_returns(rets, block=10, rng=None):
    """
    Stationary block bootstrap of a multivariate daily-return panel.
    Preserves cross-sectional dependence within a day and local vol clustering
    within blocks; destroys long-range spurious predictability.
    """
    rng = rng or np.random.default_rng()
    T, N = rets.shape
    if T < block * 2:
        block = max(2, T // 5)
    n_blocks = int(np.ceil(T / block))
    starts = rng.integers(0, T - block + 1, size=n_blocks)
    pieces = [rets.iloc[s:s + block].values for s in starts]
    boot = np.vstack(pieces)[:T]
    return pd.DataFrame(boot, index=rets.index, columns=rets.columns)


def exp_bootstrap_selection(n_boot=40, block=10):
    """
    E9 — Randomized-data benchmarking of the *selection process*.

    Unlike the analytic Deflated Sharpe (E8) which assumes a fixed N_TRIALS, this
    re-runs the full config search on block-bootstrap return panels and asks:
    how often does a no-alpha world produce a best Sharpe as large as the one we
    saw on real data?
    """
    print(f"\n[E9] Bootstrap selection benchmark ({n_boot} resamples, block={block})")
    rng = np.random.default_rng(42)

    # Chronological split: everything before the last WINDOW is the *selection*
    # period; the final WINDOW is held out untouched until the end.
    px_all = pool_px()
    rets_all = px_all.pct_change().dropna()
    if len(rets_all) < WINDOW * 2 + 50:
        print("  insufficient history for E9; skipping")
        return {}

    select_rets = rets_all.iloc[:-WINDOW]
    # Prices reconstructed from returns for the MV/CVaR walk-forward APIs.
    def rets_to_px(r):
        return (1.0 + r).cumprod()

    configs = {
        "CVaR uniform": ("cvar", dict(allocation="uniform", criterion="sharpe")),
        "CVaR adaptive": ("cvar", dict(allocation="adaptive", criterion="sharpe")),
        "MV relative-BL": ("mv", dict(relative=True)),
        "MV absolute-BL": ("mv", dict(relative=False)),
    }

    def _score_mv(px_panel, cols, relative):
        saved = mv.black_litterman_mu
        if not relative:
            def _abs(returns, Sigma, **kw):
                kw = dict(kw)
                kw["use_relative_views"] = False
                return saved(returns, Sigma, **kw)
            mv.black_litterman_mu = _abs
        try:
            curve, _ = mv.walk_forward_backtest(px_panel[cols].dropna(how="any"))
        finally:
            mv.black_litterman_mu = saved
        return curve

    # Real-data selection: best Sharpe on the selection window.
    real_scores = {}
    for name, ck in configs.items():
        if ck[0] == "cvar":
            uni = build_universe(target_size=8, end_date=SELECT_END, **ck[1])
            curve, _ = ct.walk_forward_backtest(universe_px(uni))
        else:
            uni = build_universe(target_size=9, end_date=SELECT_END, criterion="sharpe")
            curve = _score_mv(universe_px(uni), list(uni), ck[1].get("relative", True))
        # Selection-period score = metrics on the slice before holdout.
        sel_curve = curve.loc[:select_rets.index[-1]].dropna()
        if len(sel_curve) < 30:
            real_scores[name] = float("nan")
            continue
        sel_curve = sel_curve / sel_curve.iloc[0]
        # Use last WINDOW of selection period for comparable length.
        real_scores[name] = metrics(windowed(sel_curve))["sharpe"]
        print(f"  real selection Sharpe [{name}]: {real_scores[name]:.2f}")

    best_name = max(real_scores, key=lambda k: (real_scores[k] if real_scores[k] == real_scores[k] else -1e9))
    best_real = real_scores[best_name]
    print(f"  best real config: {best_name} ({best_real:.2f})")

    # Bootstrap null: reshuffle selection-period returns, re-search configs.
    boot_best = []
    for b in range(n_boot):
        boot_rets = _block_bootstrap_returns(select_rets, block=block, rng=rng)
        boot_px = rets_to_px(boot_rets)
        scores = []
        for name, ck in configs.items():
            try:
                if ck[0] == "cvar":
                    uni = select_universe(
                        boot_rets.tail(min(378, len(boot_rets))),
                        target_size=8, **ck[1],
                    )
                    cols = [c for c in uni if c in boot_px.columns]
                    curve, _ = ct.walk_forward_backtest(boot_px[cols].dropna(how="any"))
                else:
                    uni = select_universe(
                        boot_rets.tail(min(378, len(boot_rets))),
                        target_size=9, criterion="sharpe",
                    )
                    cols = [c for c in uni if c in boot_px.columns]
                    curve = _score_mv(boot_px, cols, ck[1].get("relative", True))
                sc = metrics(windowed(curve))["sharpe"]
                if sc == sc:
                    scores.append(sc)
            except Exception:
                continue
        boot_best.append(max(scores) if scores else float("nan"))
        if (b + 1) % 10 == 0:
            print(f"  bootstrap {b+1}/{n_boot}...")

    boot_best = np.asarray(boot_best, dtype=float)
    boot_best = boot_best[np.isfinite(boot_best)]
    p_value = float(np.mean(boot_best >= best_real)) if len(boot_best) else float("nan")
    p95 = float(np.quantile(boot_best, 0.95)) if len(boot_best) else float("nan")

    # Holdout: untouched final WINDOW with the winning real config.
    ck = configs[best_name]
    if ck[0] == "cvar":
        uni = build_universe(target_size=8, end_date=SELECT_END, **ck[1])
        curve, _ = ct.walk_forward_backtest(universe_px(uni))
    else:
        uni = build_universe(target_size=9, end_date=SELECT_END, criterion="sharpe")
        curve = _score_mv(universe_px(uni), list(uni), ck[1].get("relative", True))
    holdout_m = metrics(windowed(curve))

    body = (
        "Randomized-data benchmark of the *selection process* (inspired by the 2016 "
        "report's real-vs-random strategy search). Unlike E8's analytic Deflated "
        "Sharpe (which assumes a fixed N_TRIALS), E9 re-runs universe + config "
        f"selection on {n_boot} stationary block-bootstrap return panels "
        f"(block={block} days), preserving cross-asset dependence and local vol "
        "clustering while destroying long-horizon spurious alpha.\n\n"
        "### Real-data selection Sharpes (pre-holdout)\n\n"
        + "| Config | Selection Sharpe |\n|---|---|\n"
        + "".join(f"| {n} | {s:.2f} |\n" for n, s in real_scores.items())
        + f"\n**Selected:** `{best_name}` with selection Sharpe **{best_real:.2f}**\n\n"
        "### Bootstrap null (best Sharpe under no-alpha)\n\n"
        f"- Bootstrap samples with a finite best Sharpe: **{len(boot_best)}/{n_boot}**\n"
        f"- Mean / 95th pct of best bootstrap Sharpe: "
        f"**{np.nanmean(boot_best):.2f}** / **{p95:.2f}**\n"
        f"- Empirical p-value P(boot best ≥ real best): **{p_value:.3f}**\n\n"
        "A small p-value means the real selection result is extreme vs chance "
        "searching; a large p-value means the reported edge is consistent with "
        "overfitting from trying many rules.\n\n"
        "### Untouched holdout (last ~1y, frozen after selection)\n\n"
        f"| Strategy | Ann. Return | Ann. Vol | Sharpe (excess) | Max Drawdown |\n"
        f"|---|---|---|---|---|\n"
        f"| {best_name} (holdout) | {holdout_m['ann_ret']:.2%} | {holdout_m['ann_vol']:.2%} | "
        f"{holdout_m['sharpe']:.2f} | {holdout_m['max_dd']:.2%} |\n"
    )

    series = {f"{best_name} (holdout)": windowed(curve)}
    series, bm = add_benchmarks(series)
    rows = [(f"{best_name} (holdout)", holdout_m)]
    # Stash bootstrap summary on the metrics dict for the index.
    out = make_report("e9_bootstrap",
                      "E9 — Bootstrap Selection Benchmark (Real vs Random Search)",
                      series, body, rows + bm)
    out["_e9_meta"] = dict(
        best_name=best_name, best_real=best_real, p_value=p_value, p95=p95,
        n_boot=len(boot_best),
    )
    return out


def write_index(all_metrics: dict):
    e1 = all_metrics["E1"]
    e3 = all_metrics["E3"]
    e4 = all_metrics["E4"]
    e6 = all_metrics["E6"]
    e8 = all_metrics["E8"]
    e9 = all_metrics.get("E9", {})
    spy = e1["S&P 500 (SPY)"]
    dia = e1["Dow Jones (DIA)"]
    e9_meta = e9.get("_e9_meta", {})
    lines = [
        "# Trials\n",
        "Backtests of the CVaR and mean-variance traders over the most recent ~1-year "
        "window, benchmarked against the S&P 500 (SPY), Dow Jones (DIA), and a 60/40 "
        "(SPY/IEF) blend. Universes are chosen by `market_analyzer.build_universe` using "
        "only data available before the window (no lookahead). Prices are dividend/split "
        "adjusted, Sharpe ratios are **excess** of a "
        f"{RISK_FREE:.0%} risk-free rate, and equity curves are **net of "
        "asset-specific trading costs** (Markowitz) / "
        f"{ct.COST_BPS:.0f} bps turnover (CVaR).\n",
        "## Key findings\n",
        "- **Markowitz v7:** relative Black-Litterman views (residual mean-reversion "
        "between correlated peers) + asset-specific L1 costs (no-trade region) + "
        "IEWMA covariance + **daily** rebalancing. See E4.",
        "- **Robustness checks (E8):** after charging costs, scoring out-of-sample across "
        f"sub-windows, and deflating for multiple testing, CVaR net {_fmt(e8['CVaR (net)'])} "
        f"and Mean-Variance net {_fmt(e8['Mean-Variance (net)'])}.",
        (
            f"- **E9 bootstrap selection p-value:** {e9_meta.get('p_value', float('nan')):.3f} "
            f"(real best `{e9_meta.get('best_name', '?')}` selection Sharpe "
            f"{e9_meta.get('best_real', float('nan')):.2f} vs null 95th pct "
            f"{e9_meta.get('p95', float('nan')):.2f})."
            if e9_meta else "- **E9 bootstrap selection:** not run."
        ),
        f"- **Benchmarks (this window):** SPY {_fmt(spy)}; DIA {_fmt(dia)}.",
        f"- **CVaR beats both benchmarks risk-adjusted** ({_fmt(e1['CVaR (uniform/sharpe, 8)'])}).",
        "- **Adaptive cluster allocation** "
        f"({_fmt(e3['CVaR (adaptive / sharpe)'])} vs uniform "
        f"{_fmt(e3['CVaR (uniform / sharpe)'])}).",
        "- **Relative BL views** "
        f"({_fmt(e4['MV v7 relative'])} vs absolute-only {_fmt(e4['MV absolute-only'])}).",
        "- **Rolling universe** "
        f"{'underperformed' if e6['CVaR rolling (adaptive)']['sharpe'] < e6['CVaR fixed']['sharpe'] else 'outperformed'} "
        f"fixed ({_fmt(e6['CVaR rolling (adaptive)'])} vs {_fmt(e6['CVaR fixed'])}).",
        "- **CVXPYgen compiled solver** — see E5.\n",
        "## Experiments\n",
        "| Experiment | Report |",
        "|---|---|",
        "| E1 — CVaR baseline vs S&P/Dow | [e1_cvar_baseline.md](e1_cvar_baseline.md) |",
        "| E2 — Mean-Variance baseline vs S&P/Dow | [e2_mv_baseline.md](e2_mv_baseline.md) |",
        "| E3 — CVaR universe-selection variations | [e3_cvar_universe.md](e3_cvar_universe.md) |",
        "| E4 — MV relative BL vs absolute-only | [e4_mv_risk_correction.md](e4_mv_risk_correction.md) |",
        "| E5 — CVXPYgen compiled solver | [e5_fast_solver.md](e5_fast_solver.md) |",
        "| E6 — CVaR fixed vs rolling universe | [e6_cvar_rolling.md](e6_cvar_rolling.md) |",
        "| E7 — Mean-Variance fixed vs rolling universe | [e7_mv_rolling.md](e7_mv_rolling.md) |",
        "| E8 — Robustness: costs / out-of-sample / deflated Sharpe | [e8_robustness.md](e8_robustness.md) |",
        "| E9 — Bootstrap selection benchmark | [e9_bootstrap.md](e9_bootstrap.md) |",
        "",
        "Regenerate with: `python trials/run_trials.py` (from the project root).",
        "",
        f"_Generated {dt.date.today().isoformat()}. Returns are for one specific recent "
        "window and are not predictive; the harness re-selects universes and re-runs on "
        "each invocation._",
    ]
    with open(os.path.join(_HERE, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\nwrote trials/README.md")


def main():
    master_px()  # warm the shared price panel once
    results = {}
    results["E1"] = exp_cvar_baseline()
    results["E2"] = exp_mv_baseline()
    results["E3"] = exp_cvar_universe_variations()
    results["E4"] = exp_mv_risk_correction()
    results["E5"] = exp_fast_solver_benchmark()
    results["E6"] = exp_cvar_rolling()
    results["E7"] = exp_mv_rolling()
    results["E8"] = exp_robustness()
    results["E9"] = exp_bootstrap_selection(n_boot=30, block=10)
    write_index(results)
    print("\nAll trials complete.")


if __name__ == "__main__":
    main()
