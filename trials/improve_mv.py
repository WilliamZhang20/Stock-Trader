"""
A/B harness for mean-variance v7: daily rebalancing with asset-specific costs
and relative Black-Litterman views vs weekly / absolute-only ablations.

Run from project root:

    python trials/improve_mv.py
"""
from __future__ import annotations

import sys
import os
import datetime as dt

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import mean_variance_trader as mv
from market_analyzer import build_universe
from trials.run_trials import (
    universe_px, windowed, metrics, WINDOW, RISK_FREE,
    benchmark_series, benchmark_6040,
)


def backtest_variant(px, *, relative=True, rebal_every=1, use_asset_costs=True):
    """Walk-forward with temporary module knobs; restores defaults after."""
    saved = (
        mv.REBAL_EVERY_DAYS,
        mv.BASE_COST_BPS,
        mv.VOL_COST_MULT,
        mv.black_litterman_mu,
    )
    mv.REBAL_EVERY_DAYS = rebal_every
    if not use_asset_costs:
        # Approximate the old scalar L1 with a flat per-asset cost = COST_BPS.
        mv.BASE_COST_BPS = mv.COST_BPS
        mv.VOL_COST_MULT = 0.0

    _bl = saved[3]

    def _abs_only(returns, Sigma, **kwargs):
        kwargs = dict(kwargs)
        kwargs["use_relative_views"] = False
        return _bl(returns, Sigma, **kwargs)

    if not relative:
        mv.black_litterman_mu = _abs_only
    try:
        curve, weights = mv.walk_forward_backtest(px)
    finally:
        (mv.REBAL_EVERY_DAYS, mv.BASE_COST_BPS, mv.VOL_COST_MULT,
         mv.black_litterman_mu) = saved
    return curve, weights


def _turnover_stats(weights: pd.DataFrame) -> dict:
    dw = weights.diff().abs().sum(axis=1).dropna()
    # Fraction of rebalance days with almost-zero trade (no-trade region hit).
    active = dw[dw > 1e-8]
    n_rebal = max(len(active), 1)
    quiet = float((dw < 1e-4).mean()) if len(dw) else float("nan")
    return dict(
        avg_turnover=float(active.mean()) if len(active) else 0.0,
        quiet_frac=quiet,
        n_moves=int((dw > 1e-4).sum()),
        n_days=len(dw),
    )


CONFIGS = [
    dict(name="v7 daily + relative + asset c_i", relative=True, rebal_every=1, use_asset_costs=True),
    dict(name="daily + absolute + asset c_i", relative=False, rebal_every=1, use_asset_costs=True),
    dict(name="weekly + relative + asset c_i", relative=True, rebal_every=5, use_asset_costs=True),
    dict(name="daily + relative + flat 10bps", relative=True, rebal_every=1, use_asset_costs=False),
    dict(name="weekly + absolute + flat 10bps", relative=False, rebal_every=5, use_asset_costs=False),
]


def main():
    print("Fetching prices / selecting universe...")
    select_end = (dt.date.today() - dt.timedelta(days=1500)).isoformat()
    uni = build_universe(target_size=9, end_date=select_end, criterion="sharpe", verbose=False)
    print(f"Universe: {uni}")
    px = universe_px(uni)

    rows = []
    for cfg in CONFIGS:
        print(f"  running: {cfg['name']}...")
        curve, weights = backtest_variant(
            px,
            relative=cfg["relative"],
            rebal_every=cfg["rebal_every"],
            use_asset_costs=cfg["use_asset_costs"],
        )
        m = metrics(windowed(curve))
        tstat = _turnover_stats(weights.reindex(windowed(curve).index).ffill().fillna(0.0))
        rows.append((cfg["name"], m, tstat))
        print(
            f"    -> {m['ann_ret']:.2%} ret / {m['sharpe']:.2f} Sharpe / "
            f"{m['max_dd']:.2%} DD / avg L1 turn {tstat['avg_turnover']:.3f} / "
            f"quiet days {tstat['quiet_frac']:.0%}"
        )

    idx = windowed(backtest_variant(px, **{k: CONFIGS[0][k] for k in
                                           ("relative", "rebal_every", "use_asset_costs")})[0]).index
    spy = metrics(benchmark_series(idx, "SPY"))
    b60 = metrics(benchmark_6040(idx))

    print(f"\n=== Summary (last ~{WINDOW}d, excess Sharpe rf={RISK_FREE:.0%}) ===")
    print(f"{'Config':<36} {'Ret':>8} {'Vol':>8} {'Sharpe':>8} {'MaxDD':>8} {'Turn':>7} {'Quiet':>7}")
    print("-" * 90)
    for name, m, t in rows:
        print(
            f"{name:<36} {m['ann_ret']:>7.2%} {m['ann_vol']:>7.2%} "
            f"{m['sharpe']:>8.2f} {m['max_dd']:>7.2%} "
            f"{t['avg_turnover']:>7.3f} {t['quiet_frac']:>6.0%}"
        )
    print(f"{'S&P 500 (SPY)':<36} {spy['ann_ret']:>7.2%} {spy['ann_vol']:>7.2%} "
          f"{spy['sharpe']:>8.2f} {spy['max_dd']:>7.2%}")
    print(f"{'60/40 (SPY/IEF)':<36} {b60['ann_ret']:>7.2%} {b60['ann_vol']:>7.2%} "
          f"{b60['sharpe']:>8.2f} {b60['max_dd']:>7.2%}")

    best = max(rows, key=lambda x: (x[1]["sharpe"], x[1]["ann_ret"]))
    print(f"\nBest by excess Sharpe: {best[0]} "
          f"({best[1]['ann_ret']:.2%} / {best[1]['sharpe']:.2f})")


if __name__ == "__main__":
    main()
