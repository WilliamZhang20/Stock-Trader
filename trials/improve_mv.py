"""
Quick A/B harness for mean-variance performance upgrades.

Tests Practical-Markowitz knobs (IEWMA cov, weekly rebal, momentum BL views,
tighter caps, stronger trade penalty) against the current baseline on one
fixed universe / window. Run from project root:

    python trials/improve_mv.py
"""
from __future__ import annotations

import os
import sys
import copy
import math
import datetime as dt

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import mean_variance_trader as mv
from market_analyzer import build_universe
from trials.run_trials import (
    master_px, universe_px, windowed, metrics, add_benchmarks, WINDOW, RISK_FREE,
)


# ---------------------------------------------------------------------------
# Candidate estimators / knobs
# ---------------------------------------------------------------------------
def ewma_cov(returns: pd.DataFrame, halflife: float) -> pd.DataFrame:
    """Single-pass EWMA covariance (RiskMetrics-style)."""
    R = returns.values
    T, N = R.shape
    alpha = 1.0 - math.exp(-math.log(2) / halflife)
    S = np.outer(R[0], R[0])
    for t in range(1, T):
        S = (1 - alpha) * S + alpha * np.outer(R[t], R[t])
    # Ledoit-Wolf style ridge for PSD / invertibility
    S = S + 1e-8 * np.eye(N)
    return pd.DataFrame(S, index=returns.columns, columns=returns.columns)


def iewma_cov(returns: pd.DataFrame, vol_halflife: float = 63, cor_halflife: float = 125) -> pd.DataFrame:
    """
    Iterated EWMA (Engle / Barratt-Boyd): EWMA vols, then EWMA correlation of
    volatility-standardized returns. Far more responsive than Ledoit-Wolf on a
    flat trailing window.
    """
    R = returns.values.astype(float)
    T, N = R.shape
    a_vol = 1.0 - math.exp(-math.log(2) / vol_halflife)
    a_cor = 1.0 - math.exp(-math.log(2) / cor_halflife)

    # Step 1: EWMA variance per asset
    var = np.maximum(R[0] ** 2, 1e-8)
    vols = np.zeros((T, N))
    vols[0] = np.sqrt(var)
    for t in range(1, T):
        var = (1 - a_vol) * var + a_vol * R[t] ** 2
        vols[t] = np.sqrt(np.maximum(var, 1e-8))

    # Step 2: EWMA correlation of standardized returns
    Z = R / vols
    C = np.outer(Z[0], Z[0])
    for t in range(1, T):
        C = (1 - a_cor) * C + a_cor * np.outer(Z[t], Z[t])
    # Force unit diagonal, then rebuild Sigma = D C D
    d = np.sqrt(np.clip(np.diag(C), 1e-8, None))
    C = C / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    # Clip eigenvalues for PSD
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.clip(eigvals, 1e-6, None)
    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Renormalize diagonal after clip
    d = np.sqrt(np.clip(np.diag(C), 1e-8, None))
    C = C / np.outer(d, d)
    np.fill_diagonal(C, 1.0)

    D = np.diag(vols[-1])
    S = D @ C @ D + 1e-8 * np.eye(N)
    return pd.DataFrame(S, index=returns.columns, columns=returns.columns)


def momentum_views(returns: pd.DataFrame, lookback: int = 63) -> pd.Series:
    """Average daily return over a medium-term momentum window (skipping last 5 days to reduce reversal)."""
    lb = min(lookback, len(returns) - 5)
    if lb < 10:
        return returns.mean()
    window = returns.iloc[-(lb + 5):-5] if len(returns) > lb + 5 else returns.iloc[-lb:]
    return window.mean()


def black_litterman_mu_custom(returns, Sigma, view_series, tau=0.05, delta=2.5, view_unc=0.25):
    """BL posterior with caller-supplied views and tunable view uncertainty."""
    from black_litterman import black_litterman_posterior
    syms = list(returns.columns)
    n = len(syms)
    Sig = Sigma.values
    vol = np.sqrt(np.clip(np.diag(Sig), 1e-12, None))
    w_mkt = (1.0 / vol)
    w_mkt = w_mkt / w_mkt.sum()
    w_mkt_scaled = w_mkt * (delta / tau)
    q = view_series.reindex(syms).fillna(0.0).values
    P = np.eye(n)
    omega_diag = np.clip(np.diag(P @ (tau * Sig) @ P.T), 1e-12, None) * view_unc
    Omega = np.diag(omega_diag)
    try:
        pi = black_litterman_posterior(Sig, w_mkt_scaled, P, q, Omega, tau=tau)
    except np.linalg.LinAlgError:
        pi = q
    return pd.Series(np.asarray(pi).reshape(-1), index=syms)


def solve_with_cfg(mu, Sigma, w_prev, cfg):
    """Solve using temporary module-level knobs from cfg."""
    saved = (mv.W_MAX, mv.GAMMA_TC, mv.TAU_TURNOVER, mv.LAMBDA_RISK, mv.TARGET_VOL, mv.MAX_INVEST)
    mv.W_MAX = cfg["w_max"]
    mv.GAMMA_TC = cfg["gamma_tc"]
    mv.TAU_TURNOVER = cfg["tau_turnover"]
    mv.LAMBDA_RISK = cfg["lambda_risk"]
    mv.TARGET_VOL = cfg["target_vol"]
    try:
        return mv.solve_portfolio(
            mu, Sigma, w_prev=w_prev,
            max_invest_fraction=mv.MAX_INVEST,
            lambda_risk=cfg["lambda_risk"],
            target_vol=cfg["target_vol"],
        )
    finally:
        mv.W_MAX, mv.GAMMA_TC, mv.TAU_TURNOVER, mv.LAMBDA_RISK, mv.TARGET_VOL, mv.MAX_INVEST = saved


def backtest_cfg(px, cfg, cost_bps=10.0):
    """Walk-forward backtest with a config dict controlling all knobs."""
    rets = px.pct_change().dropna()
    dates = rets.index
    lookback = cfg["lookback"]
    rebal_every = cfg["rebal_every"]  # trading days between rebalances

    current_w = pd.Series(0.0, index=px.columns)
    held_w = pd.Series(0.0, index=px.columns)
    w_prev = None
    equity = 1.0
    curve = []
    last_rebal = -10 ** 9

    for t_idx, today in enumerate(dates):
        if t_idx >= lookback and (t_idx - last_rebal) >= rebal_every:
            window = rets.iloc[t_idx - lookback:t_idx]
            # Covariance
            if cfg["cov"] == "lw":
                Sigma = mv.shrinkage_cov(window)
            elif cfg["cov"] == "ewma":
                Sigma = ewma_cov(window, cfg["cov_halflife"])
            else:
                Sigma = iewma_cov(window, cfg["vol_hl"], cfg["cor_hl"])
            # Expected returns
            if cfg["mu"] == "bl_ewma":
                mu = mv.black_litterman_mu(window, Sigma, halflife_days=cfg["mu_hl"],
                                           view_uncertainty=cfg["view_unc"])
            elif cfg["mu"] == "bl_mom":
                views = momentum_views(window, cfg["mom_lb"])
                mu = black_litterman_mu_custom(window, Sigma, views, view_unc=cfg["view_unc"])
            elif cfg["mu"] == "ewma":
                mu = mv.exp_weighted_mean_returns(window, cfg["mu_hl"])
            else:  # mom
                mu = momentum_views(window, cfg["mom_lb"])
            current_w = solve_with_cfg(mu, Sigma, w_prev, cfg)
            w_prev = current_w.values.copy()
            last_rebal = t_idx

        turnover = float((current_w - held_w).abs().sum())
        if turnover > 0:
            equity *= (1.0 - cost_bps / 1e4 * turnover)
        if t_idx > 0:
            equity *= 1.0 + float((rets.iloc[t_idx] * current_w).sum())
        held_w = current_w.copy()
        curve.append((today, equity))

    return pd.Series(dict(curve)).sort_index().rename("Equity")


# ---------------------------------------------------------------------------
# Configs to try
# ---------------------------------------------------------------------------
BASE = dict(
    name="baseline (current)",
    cov="lw", mu="bl_ewma", mu_hl=5, view_unc=1.0, mom_lb=63,
    lookback=100, rebal_every=1, cov_halflife=60, vol_hl=63, cor_hl=125,
    w_max=0.40, lambda_risk=7.0, gamma_tc=0.001, tau_turnover=0.40, target_vol=0.12,
)

CONFIGS = [
    BASE,
    # 1. Just weekly rebal (cut cost drag)
    {**BASE, "name": "weekly rebal", "rebal_every": 5},
    # 2. IEWMA cov + weekly
    {**BASE, "name": "IEWMA + weekly", "cov": "iewma", "rebal_every": 5, "lookback": 252},
    # 3. Medium-term momentum views, less BL shrinkage
    {**BASE, "name": "BL-mom + weekly", "mu": "bl_mom", "view_unc": 0.25, "rebal_every": 5,
     "lookback": 252, "mom_lb": 63},
    # 4. Full practical stack (IEWMA + mom views + weekly + tighter caps from E4)
    {**BASE, "name": "practical stack", "cov": "iewma", "mu": "bl_mom", "view_unc": 0.25,
     "rebal_every": 5, "lookback": 252, "mom_lb": 63,
     "w_max": 0.25, "lambda_risk": 10.0, "gamma_tc": 0.005, "tau_turnover": 0.25, "target_vol": 0.14},
    # 5. Aggressive practical (more return weight, higher vol target)
    {**BASE, "name": "aggressive practical", "cov": "iewma", "mu": "bl_mom", "view_unc": 0.15,
     "rebal_every": 5, "lookback": 252, "mom_lb": 126,
     "w_max": 0.30, "lambda_risk": 5.0, "gamma_tc": 0.003, "tau_turnover": 0.30, "target_vol": 0.16},
    # 6. Raw momentum + IEWMA (no BL) — see if BL is the bottleneck
    {**BASE, "name": "raw-mom + IEWMA", "cov": "iewma", "mu": "mom", "rebal_every": 5,
     "lookback": 252, "mom_lb": 63,
     "w_max": 0.25, "lambda_risk": 8.0, "gamma_tc": 0.005, "tau_turnover": 0.25, "target_vol": 0.14},
    # 7. E4-style tightened defaults on current estimator
    {**BASE, "name": "E4 tightened", "w_max": 0.20, "lambda_risk": 15.0, "rebal_every": 5, "target_vol": 0.12},
]


def main():
    print("Fetching prices / selecting universe...")
    # Match the trials harness selection date so results are comparable.
    select_end = (dt.date.today() - dt.timedelta(days=1500)).isoformat()
    uni = build_universe(target_size=9, end_date=select_end, criterion="sharpe", verbose=False)
    print(f"Universe: {uni}")
    px = universe_px(uni)

    rows = []
    for cfg in CONFIGS:
        print(f"  running: {cfg['name']}...")
        curve = backtest_cfg(px, cfg)
        m = metrics(windowed(curve))
        rows.append((cfg["name"], m))
        print(f"    -> {m['ann_ret']:.2%} ret / {m['sharpe']:.2f} Sharpe / {m['max_dd']:.2%} DD / vol {m['ann_vol']:.2%}")

    # Benchmarks
    series = {name: windowed(backtest_cfg(px, next(c for c in CONFIGS if c["name"] == name)))
              for name, _ in rows[:1]}  # just for index
    # cheaper: rebuild index from last curve
    last_curve = windowed(backtest_cfg(px, CONFIGS[0]))
    from trials.run_trials import benchmark_series, benchmark_6040
    idx = last_curve.index
    spy = metrics(benchmark_series(idx, "SPY"))
    b60 = metrics(benchmark_6040(idx))

    print("\n=== Summary (last ~1y, net of 10bps, excess Sharpe) ===")
    print(f"{'Config':<28} {'Ret':>8} {'Vol':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print("-" * 64)
    for name, m in rows:
        print(f"{name:<28} {m['ann_ret']:>7.2%} {m['ann_vol']:>7.2%} {m['sharpe']:>8.2f} {m['max_dd']:>7.2%}")
    print(f"{'S&P 500 (SPY)':<28} {spy['ann_ret']:>7.2%} {spy['ann_vol']:>7.2%} {spy['sharpe']:>8.2f} {spy['max_dd']:>7.2%}")
    print(f"{'60/40 (SPY/IEF)':<28} {b60['ann_ret']:>7.2%} {b60['ann_vol']:>7.2%} {b60['sharpe']:>8.2f} {b60['max_dd']:>7.2%}")

    # Pick best by Sharpe, then return
    best = max(rows, key=lambda x: (x[1]["sharpe"], x[1]["ann_ret"]))
    print(f"\nBest by excess Sharpe: {best[0]} "
          f"({best[1]['ann_ret']:.2%} / {best[1]['sharpe']:.2f})")


if __name__ == "__main__":
    main()
