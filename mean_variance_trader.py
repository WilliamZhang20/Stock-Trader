#!/usr/bin/env python3
"""
Version 5: rolling mean-variance (Markowitz) portfolio optimization.

Two upgrades over the textbook version, which is notoriously unstable because a
raw sample mean is a very noisy return estimate:

1. Expected returns come from a **Black-Litterman posterior** (a market-equilibrium
   prior shrinks the noisy EWMA/momentum "views"), not a raw sample mean. This is
   the dominant lever for out-of-sample robustness.
2. Portfolio risk is stabilized with an explicit **volatility target** rather than
   a fragile regime-switching HMM. One interpretable knob (TARGET_VOL) scales gross
   exposure so realized risk stays roughly constant across regimes.

Together with the turnover penalty this is the "practical Markowitz" setup from
https://stanford.edu/class/engr108/lectures/portfolio_slides.pdf
"""
import os, argparse, math, datetime as dt
import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.covariance import LedoitWolf

# Alpaca SDK
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from market_analyzer import build_universe, select_universe
from black_litterman import black_litterman_posterior

# Total-return (splits + dividends) adjustment for fair, leak-free prices.
try:
    from alpaca.data.enums import Adjustment
    _ADJ_ALL = Adjustment.ALL
except Exception:
    _ADJ_ALL = "all"

START = "2023-01-01"
END   = None
REBAL_FREQ = "B" # Daily
RETURN_LOOKBACK_DAYS = 100
EWMA_HALFLIFE_DAYS = 5
LAMBDA_RISK = 7.0
GAMMA_TC = 0.001
TAU_TURNOVER = 0.40
W_MAX = 0.40 # Slightly relaxed max weight
MAX_INVEST = 0.99

# Black-Litterman parameters (return-estimate shrinkage).
BL_TAU = 0.05              # uncertainty of the equilibrium prior
BL_DELTA = 2.5             # market risk-aversion for reverse optimization
BL_VIEW_UNCERTAINTY = 1.0  # scales Omega; higher => trust the prior more (more shrinkage)

# Volatility targeting (replaces the old HMM regime multipliers).
TARGET_VOL = 0.12          # annualized portfolio volatility target

# Transaction costs charged in the backtest (round-trip, bps of traded notional).
COST_BPS = 10.0

# -----------------------
# Alpaca helpers
# -----------------------
def fetch_alpaca_prices(symbols, start, end):
    """Fetch daily total-return (adjusted) close bars from Alpaca Data API."""
    key = os.environ["APCA_API_KEY_ID"]
    secret = os.environ["APCA_API_SECRET_KEY"]
    client = StockHistoricalDataClient(key, secret)

    if end is None:
        end = dt.date.today()

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=pd.to_datetime(start),
        end=pd.to_datetime(end),
        adjustment=_ADJ_ALL,
    )
    bars = client.get_stock_bars(request).df

    px = bars["close"].unstack(level=0)
    return px.sort_index()

def rebalance_alpaca_to_weights(target_w, notional):
    """Send market orders to reach target weights."""
    key = os.environ["APCA_API_KEY_ID"]
    secret = os.environ["APCA_API_SECRET_KEY"]
    trading_client = TradingClient(key, secret, paper=True)

    current_positions = {p.symbol: float(p.market_value) for p in trading_client.get_all_positions()}

    for sym, target_w in target_w.items():
        target_notional = target_w * notional
        current_notional = current_positions.get(sym, 0.0)
        delta = target_notional - current_notional
        if abs(delta) < 1.0:
            continue
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        rounded_notional = round(abs(delta), 2)
        if rounded_notional < 1.0:
            continue
        req = MarketOrderRequest(
            symbol=sym,
            notional=rounded_notional,
            side=side,
            time_in_force=TimeInForce.DAY
        )
        trading_client.submit_order(req)

# -----------------------
# Return / risk estimation
# -----------------------
def exp_weighted_mean_returns(returns, halflife_days):
    lam = math.log(2)/halflife_days
    w = np.exp(-lam * np.arange(len(returns))[::-1])
    w = w / w.sum()
    mu = (returns * w[:,None]).sum(axis=0)
    return pd.Series(mu, index=returns.columns)

def shrinkage_cov(returns):
    lw = LedoitWolf().fit(returns.values)
    return pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)

def black_litterman_mu(
    returns,
    Sigma,
    halflife_days=EWMA_HALFLIFE_DAYS,
    tau=BL_TAU,
    delta=BL_DELTA,
    view_uncertainty=BL_VIEW_UNCERTAINTY,
):
    """
    Black-Litterman posterior expected returns, used as `mu` for the QP.

    The noisy EWMA/momentum estimate enters only as *views*; they are shrunk
    toward a market-equilibrium prior obtained by reverse optimization from an
    inverse-volatility market proxy. This dramatically reduces the estimation
    error that makes plain Markowitz overfit.

    Wires in the standalone `black_litterman.py` module. That helper builds its
    internal prior as ``pi = tau * Sigma @ w_mkt``; we pre-scale the market
    weights by ``delta / tau`` so the effective prior is the standard
    reverse-optimization equilibrium ``pi = delta * Sigma @ w_mkt``.
    """
    syms = list(returns.columns)
    n = len(syms)
    Sig = Sigma.values

    # Inverse-volatility market proxy (normalized to sum to 1).
    vol = np.sqrt(np.clip(np.diag(Sig), 1e-12, None))
    w_mkt = 1.0 / vol
    w_mkt = w_mkt / w_mkt.sum()
    w_mkt_scaled = w_mkt * (delta / tau)

    # One view per asset: its EWMA/momentum expected return.
    q = exp_weighted_mean_returns(returns, halflife_days).values
    P = np.eye(n)

    # He-Litterman view uncertainty: proportional to prior variance of each view.
    omega_diag = np.clip(np.diag(P @ (tau * Sig) @ P.T), 1e-12, None) * view_uncertainty
    Omega = np.diag(omega_diag)

    try:
        pi_bl = black_litterman_posterior(Sig, w_mkt_scaled, P, q, Omega, tau=tau)
    except np.linalg.LinAlgError:
        pi_bl = q  # singular posterior -> fall back to the raw views
    return pd.Series(np.asarray(pi_bl).reshape(-1), index=syms)

# -----------------------
# Optimizer
# -----------------------
def _apply_vol_target(w, Sigma, target_vol, max_invest, w_max):
    """
    Scale gross exposure so the forecast annualized portfolio volatility hits
    `target_vol`, never breaching the budget (`max_invest`) or per-name (`w_max`)
    caps. This is the transparent replacement for the old HMM risk multipliers:
    lever toward the caps when markets are calm, hold cash when they are volatile.
    """
    wv = w.values
    var = float(wv @ Sigma.values @ wv)
    ann_vol = math.sqrt(max(var, 0.0)) * math.sqrt(252)
    if ann_vol <= 1e-8:
        return w
    s = target_vol / ann_vol
    total = float(w.sum())
    if total > 0:
        s = min(s, max_invest / total)
    mx = float(w.max())
    if mx > 0:
        s = min(s, w_max / mx)
    return w * max(s, 0.0)

def solve_portfolio(
    mu,
    Sigma,
    w_prev=None,
    max_invest_fraction=MAX_INVEST,
    lambda_risk=LAMBDA_RISK,
    target_vol=TARGET_VOL,
):
    """
    Solve mean-variance portfolio with turnover regularization (least-squares
    risk term), then apply volatility targeting to the solution.
    """
    n = len(mu)
    w = cp.Variable(n)

    L = np.linalg.cholesky(Sigma.values + 1e-8 * np.eye(n))  # jitter for numerical stability

    obj = mu.values @ w - lambda_risk * cp.sum_squares(L.T @ w)

    constraints = [cp.sum(w) <= max_invest_fraction,
                   w >= 0,
                   w <= W_MAX]

    if w_prev is None:
        w_prev = np.zeros(n)
    turnover = cp.norm1(w - w_prev)
    obj = obj - GAMMA_TC * turnover
    constraints.append(turnover <= TAU_TURNOVER)

    prob = cp.Problem(cp.Maximize(obj), constraints)
    prob.solve(solver=cp.OSQP, verbose=False, max_iter=10000)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"Optimizer warning: {prob.status}")

    w_opt = pd.Series(np.clip(w.value if w.value is not None else np.zeros(n), 0, 1), index=mu.index)
    total_alloc = w_opt.sum()
    if total_alloc > max_invest_fraction:
        w_opt = w_opt * (max_invest_fraction / total_alloc)

    if target_vol and target_vol > 0:
        w_opt = _apply_vol_target(w_opt, Sigma, target_vol, max_invest_fraction, W_MAX)

    return w_opt

# Run Backtest
def walk_forward_backtest(px):
    rets = px.pct_change().dropna()
    dates = rets.index
    rebal_dates = pd.date_range(dates[0], dates[-1], freq=REBAL_FREQ)
    rebal_dates = [d for d in rebal_dates if d in dates and d >= dates[RETURN_LOOKBACK_DAYS]]

    w_prev = None
    current_w = pd.Series(0.0, index=px.columns)
    held_w = pd.Series(0.0, index=px.columns)   # weights held into each day (for cost accounting)
    equity = 1.0
    equity_curve = []
    weights_record = {}

    for t_idx, today in enumerate(dates):
        if today in rebal_dates:
            # No lookahead: trade using only returns STRICTLY before `today`.
            window = rets.iloc[t_idx - RETURN_LOOKBACK_DAYS:t_idx]
            assert window.index[-1] < today, "lookahead: window must end before the trade day"
            Sigma = shrinkage_cov(window)
            mu = black_litterman_mu(window, Sigma)
            current_w = solve_portfolio(mu, Sigma, w_prev, lambda_risk=LAMBDA_RISK)
            weights_record[today] = current_w
            w_prev = current_w.values.copy()

        # Charge transaction costs on the change from yesterday's held weights.
        turnover = float((current_w - held_w).abs().sum())
        if turnover > 0:
            equity *= (1.0 - COST_BPS / 1e4 * turnover)

        if t_idx > 0:
            day_ret = float((rets.iloc[t_idx] * current_w).sum())
            equity *= (1.0 + day_ret)
        held_w = current_w.copy()
        equity_curve.append((today, equity))

    curve = pd.Series(dict(equity_curve)).sort_index().rename("Equity")
    weights_panel = pd.DataFrame(weights_record).T.reindex(curve.index).ffill().fillna(0.0)
    return curve, weights_panel


def walk_forward_backtest_dynamic(
    pool_px,
    target_size=9,
    universe_refresh_days=63,
    select_lookback=378,
    select_kwargs=None,
    lambda_risk=LAMBDA_RISK,
):
    """
    Mean-variance walk-forward backtest with a ROLLING universe.

    Every `universe_refresh_days`, the candidate pool is re-surveyed (using only
    past data) so the portfolio rotates into newly-leading assets; with adaptive
    `select_kwargs` the per-cluster slot weighting re-adapts to recent performance
    at each refresh. Uses Black-Litterman returns + volatility targeting like the
    fixed-universe path.

    Returns (curve, weights, universe_log).
    """
    select_kwargs = dict(select_kwargs or {})
    rets = pool_px.pct_change().dropna()
    dates = rets.index

    w = pd.Series(0.0, index=pool_px.columns)
    held_w = pd.Series(0.0, index=pool_px.columns)
    universe = None
    last_uni_refresh = -10 ** 9
    equity = 1.0
    equity_curve = []
    weights_record = {}
    universe_log = {}

    for t_idx, today in enumerate(dates):
        if t_idx >= RETURN_LOOKBACK_DAYS:
            if universe is None or (t_idx - last_uni_refresh) >= universe_refresh_days:
                hist = rets.iloc[max(0, t_idx - select_lookback):t_idx]
                try:
                    new_universe = select_universe(hist, target_size=target_size, **select_kwargs)
                except Exception as e:
                    print(f"Universe selection failed at {today.date()}: {e}")
                    new_universe = universe
                if new_universe and new_universe != universe:
                    universe = new_universe
                    universe_log[today] = list(universe)
                last_uni_refresh = t_idx

            window = rets.iloc[t_idx - RETURN_LOOKBACK_DAYS:t_idx][universe]
            Sigma = shrinkage_cov(window)
            mu = black_litterman_mu(window, Sigma)
            cur = solve_portfolio(mu, Sigma, w_prev=w[universe].values, lambda_risk=lambda_risk)

            w = pd.Series(0.0, index=pool_px.columns)
            w[universe] = cur.values
            weights_record[today] = w.copy()

        turnover = float((w - held_w).abs().sum())
        if turnover > 0:
            equity *= (1.0 - COST_BPS / 1e4 * turnover)

        if t_idx > 0 and w.sum() > 0:
            equity *= 1.0 + float((rets.iloc[t_idx] * w).sum())
        held_w = w.copy()
        equity_curve.append((today, equity))

    curve = pd.Series(dict(equity_curve)).sort_index().rename("Equity")
    weights_panel = pd.DataFrame(weights_record).T.reindex(curve.index).ffill().fillna(0.0)
    return curve, weights_panel, universe_log

# -----------------------
# CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--notional", type=float, default=10000.0)
    parser.add_argument("--universe-size", type=int, default=9, help="Number of assets to select (default: 9)")
    parser.add_argument(
        "--universe-criterion", default="sharpe", choices=["sharpe", "calmar", "centroid"],
        help="Selection criterion within each cluster (default: sharpe)"
    )
    args = parser.parse_args()

    universe_end = START if args.backtest else None
    print(f"Selecting universe (size={args.universe_size}, criterion={args.universe_criterion})...")
    universe = build_universe(
        target_size=args.universe_size,
        criterion=args.universe_criterion,
        end_date=universe_end,
        verbose=True,
    )
    print()

    px = fetch_alpaca_prices(universe, start=START, end=END)

    if args.backtest:
        curve, weights = walk_forward_backtest(px)
        ann_ret = (curve.iloc[-1] / curve.iloc[0]) ** (252/len(curve)) - 1
        daily_rets = curve.pct_change().dropna()
        ann_vol = daily_rets.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        # Maximum drawdown (peak-to-trough)
        running_max = curve.cummax()
        drawdown = (curve - running_max) / running_max
        max_drawdown = drawdown.min()
        print(f"Backtest (net of {COST_BPS:.0f}bps costs): Return {ann_ret:.2%}, "
              f"Volatility {ann_vol:.2%}, Sharpe {sharpe:.2f}, Max Drawdown {max_drawdown:.2%}")

    if args.paper:
        key = os.environ["APCA_API_KEY_ID"]
        secret = os.environ["APCA_API_SECRET_KEY"]
        trading_client = TradingClient(key, secret, paper=True)
        account = trading_client.get_account()
        equity = float(account.equity)
        current_positions = {p.symbol: float(p.market_value) for p in trading_client.get_all_positions() if p.symbol in universe}
        current_w = pd.Series({sym: current_positions.get(sym, 0.0) / equity for sym in universe}) if equity > 0 else pd.Series(0.0, index=universe)

        rets = px.pct_change().dropna()
        window = rets.tail(RETURN_LOOKBACK_DAYS)

        Sigma = shrinkage_cov(window)
        mu = black_litterman_mu(window, Sigma)
        w = solve_portfolio(mu, Sigma, w_prev=current_w.values, lambda_risk=LAMBDA_RISK)
        print("Target weights:\n", w.round(4))
        rebalance_alpaca_to_weights(w, notional=equity)

if __name__ == "__main__":
    main()
