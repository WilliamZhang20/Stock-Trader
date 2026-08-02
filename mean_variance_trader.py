#!/usr/bin/env python3
"""
Version 7: practical Markowitz with relative Black-Litterman views and
asset-specific trading costs (no-trade region).

Key upgrades vs textbook / v6:
1. **Black-Litterman posterior** with *relative* views from residual
   mean-reversion between correlated peers (not just P = I absolute views).
2. **IEWMA covariance** (Engle / Barratt-Boyd).
3. **Asset-specific L1 costs** ``sum_i c_i |w_i - w_prev_i|`` in the QP *and*
   the backtest — creates a genuine no-trade region so daily rebalancing no
   longer implies daily churn.
4. **Volatility targeting** for stable risk.
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
# Daily rebalance is fine once asset-specific costs create a no-trade region;
# the optimizer simply skips names whose expected edge < marginal cost.
REBAL_EVERY_DAYS = 1
RETURN_LOOKBACK_DAYS = 252
EWMA_HALFLIFE_DAYS = 5
LAMBDA_RISK = 12.0
W_MAX = 0.25
MAX_INVEST = 0.99

# Black-Litterman parameters.
BL_TAU = 0.05
BL_DELTA = 2.5
BL_VIEW_UNCERTAINTY = 1.0          # scales absolute-view Omega
BL_REL_VIEW_UNCERTAINTY = 1.0      # base scale for relative-view Omega
BL_REL_MAX_PAIRS = 6               # cap number of relative views
BL_REL_MIN_CORR = 0.55             # only pair assets at least this correlated
BL_REL_HALFLIFE = 63               # rolling window for beta / residual
BL_REL_KAPPA = 0.25                # fraction of residual expected to mean-revert / day

# IEWMA covariance half-lives.
IEWMA_VOL_HALFLIFE = 63
IEWMA_COR_HALFLIFE = 125

# Volatility targeting.
TARGET_VOL = 0.14

# Asset-specific trading costs (one-way, as a fraction of NAV per unit weight).
# Backtest and optimizer MUST use the same c_i. Floor = BASE_COST_BPS;
# scales with recent vol as a spread/impact proxy.
BASE_COST_BPS = 5.0          # half-spread + fee floor (one-way)
VOL_COST_MULT = 0.35         # extra bps ≈ VOL_COST_MULT * (ann_vol * 100)
COST_BPS = 10.0              # legacy scalar fallback (≈ 2 * BASE for reporting)

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
    """Ledoit-Wolf shrinkage covariance (kept as a fallback / ablation)."""
    lw = LedoitWolf().fit(returns.values)
    return pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)


def iewma_cov(returns, vol_halflife=IEWMA_VOL_HALFLIFE, cor_halflife=IEWMA_COR_HALFLIFE):
    """
    Iterated EWMA covariance (Engle / Barratt-Boyd):

      1. EWMA variance → per-asset volatility forecast
      2. Standardize returns by those vols
      3. EWMA correlation of the standardized series
      4. Rebuild Sigma = D C D

    Far more responsive to regime shifts than Ledoit-Wolf on a flat window —
    this was the single largest performance lever in trials/improve_mv.py.
    """
    R = returns.values.astype(float)
    T, N = R.shape
    if T < 5:
        return shrinkage_cov(returns)

    a_vol = 1.0 - math.exp(-math.log(2) / vol_halflife)
    a_cor = 1.0 - math.exp(-math.log(2) / cor_halflife)

    var = np.maximum(R[0] ** 2, 1e-8)
    vols = np.zeros((T, N))
    vols[0] = np.sqrt(var)
    for t in range(1, T):
        var = (1 - a_vol) * var + a_vol * R[t] ** 2
        vols[t] = np.sqrt(np.maximum(var, 1e-8))

    Z = R / vols
    C = np.outer(Z[0], Z[0])
    for t in range(1, T):
        C = (1 - a_cor) * C + a_cor * np.outer(Z[t], Z[t])

    # Force a valid correlation matrix (unit diagonal + PSD).
    d = np.sqrt(np.clip(np.diag(C), 1e-8, None))
    C = C / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.clip(eigvals, 1e-6, None)
    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.clip(np.diag(C), 1e-8, None))
    C = C / np.outer(d, d)
    np.fill_diagonal(C, 1.0)

    D = np.diag(vols[-1])
    S = D @ C @ D + 1e-8 * np.eye(N)
    return pd.DataFrame(S, index=returns.columns, columns=returns.columns)


def estimate_cov(returns):
    """Default covariance estimator used by the live / backtest paths."""
    return iewma_cov(returns)


def estimate_trading_costs(returns, base_bps=BASE_COST_BPS, vol_mult=VOL_COST_MULT):
    """
    Per-asset one-way trading cost as a fraction of NAV per unit weight change.

    c_i = (base_bps + vol_mult * ann_vol_i_pct) / 1e4

    Higher-vol names pay a wider effective spread / impact. Used identically in
    the QP (``c @ |dw|``) and the equity-curve backtest so the optimizer and
    evaluation solve the same economic problem. A properly calibrated L1 cost
    induces a no-trade region: the solver only moves w_i when expected benefit
    exceeds c_i.
    """
    # Recent realized vol (prefer last ~63d when available).
    window = returns.tail(min(63, len(returns)))
    ann_vol = window.std() * np.sqrt(252)
    # ann_vol is a fraction (e.g. 0.20); convert to percent points for the mult.
    c_bps = base_bps + vol_mult * (ann_vol * 100.0)
    c = (c_bps / 1e4).clip(lower=base_bps / 1e4)
    return pd.Series(c, index=returns.columns)


def _pca_cluster_labels(returns, n_clusters=None):
    """
    Soft mirror of market_analyzer's PCA + KMeans grouping, applied to the
    *current* portfolio so relative views prefer within-cluster peers.
    """
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    R = returns.dropna(axis=1)
    n = R.shape[1]
    if n < 3:
        return {s: 0 for s in returns.columns}
    if n_clusters is None:
        n_clusters = max(2, min(n // 2, 4))
    n_clusters = max(1, min(n_clusters, n))
    # Standardize columns; fall back to zeros on degenerate vols.
    Z = R.values.astype(float)
    mu = Z.mean(axis=0)
    sd = Z.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Z = (Z - mu) / sd
    n_comp = min(n_clusters, n, max(1, Z.shape[0] // 5))
    try:
        loadings = PCA(n_components=n_comp).fit_transform(Z.T)  # assets × factors
        labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(loadings)
    except Exception:
        return {s: 0 for s in returns.columns}
    return {sym: int(lab) for sym, lab in zip(R.columns, labels)}


def _relative_pair_views(returns, Sigma, max_pairs=BL_REL_MAX_PAIRS,
                         min_corr=BL_REL_MIN_CORR, lookback=BL_REL_HALFLIFE,
                         kappa=BL_REL_KAPPA):
    """
    Build relative Black-Litterman views from residual mean-reversion.

    For correlated peers (i, j), fit r_i ≈ α + β r_j on a trailing window.
    If the residual looks mean-reverting (AR(1) |φ| < 1 with decent fit),
    emit a relative view P_k = e_i - β e_j with q_k = -κ * last_residual,
    and Ω_kk large when the relationship is unstable (low R² / high resid vol).

    Pair ranking prefers PCA/KMeans cluster mates (same idea as
    ``market_analyzer`` universe construction) before raw correlation.

    Returns (P, q, omega_diag) possibly empty (0 rows).
    """
    syms = list(returns.columns)
    n = len(syms)
    if n < 2 or len(returns) < max(40, lookback // 2):
        return np.zeros((0, n)), np.zeros(0), np.zeros(0)

    window = returns.tail(min(lookback, len(returns)))
    R = window.values
    # Correlation for pair ranking (use Sigma-implied corr when possible).
    vol = np.sqrt(np.clip(np.diag(Sigma.values), 1e-12, None))
    corr = Sigma.values / np.outer(vol, vol)
    np.fill_diagonal(corr, 0.0)

    cluster = _pca_cluster_labels(window)
    # Rank pairs: intra-cluster first, then by |corr| descending.
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            c = abs(corr[i, j])
            if c >= min_corr:
                same = 1 if cluster.get(syms[i], -1) == cluster.get(syms[j], -2) else 0
                pairs.append((same, c, i, j))
    pairs.sort(reverse=True)

    P_rows, q_list, om_list = [], [], []
    used = set()
    for _same, c, i, j in pairs:
        if len(P_rows) >= max_pairs:
            break
        # Prefer each name in at most one relative view (keeps P well-conditioned).
        if i in used or j in used:
            continue

        yi, yj = R[:, i], R[:, j]
        # OLS through demeaned series: β = cov(i,j)/var(j)
        var_j = float(np.var(yj))
        if var_j < 1e-16:
            continue
        beta = float(np.cov(yi, yj, ddof=0)[0, 1] / var_j)
        alpha = float(yi.mean() - beta * yj.mean())
        resid = yi - alpha - beta * yj
        # AR(1) on residual: φ = corr(e_t, e_{t-1})
        if len(resid) < 20:
            continue
        e0, e1 = resid[1:], resid[:-1]
        denom = float(np.dot(e1, e1))
        if denom < 1e-16:
            continue
        phi = float(np.dot(e0, e1) / denom)
        # Require mean-reversion (φ < 1) and not explosive negative.
        if not (-0.2 < phi < 0.95):
            continue
        ss_tot = float(np.dot(yi - yi.mean(), yi - yi.mean())) + 1e-16
        ss_res = float(np.dot(resid, resid))
        r2 = max(0.0, 1.0 - ss_res / ss_tot)
        if r2 < 0.15:
            continue

        # Expected relative return: residual expected to decay toward 0.
        # q ≈ -κ * (1-φ) * e_t  (speed of mean reversion times current gap)
        speed = max(1.0 - phi, 0.05)
        q_k = -kappa * speed * float(resid[-1])

        row = np.zeros(n)
        row[i] = 1.0
        row[j] = -beta
        # View uncertainty: larger when R² is low or residual is noisy.
        # Scale relative to prior variance of the view: row @ (τ Σ) @ row.
        prior_var = float(row @ (BL_TAU * Sigma.values) @ row)
        prior_var = max(prior_var, 1e-12)
        instability = (1.0 - r2) + (ss_res / len(resid)) / (np.var(yi) + 1e-12)
        omega_k = prior_var * BL_REL_VIEW_UNCERTAINTY * (1.0 + 3.0 * instability)

        P_rows.append(row)
        q_list.append(q_k)
        om_list.append(omega_k)
        used.add(i)
        used.add(j)

    if not P_rows:
        return np.zeros((0, n)), np.zeros(0), np.zeros(0)
    return np.asarray(P_rows), np.asarray(q_list), np.asarray(om_list)


def black_litterman_mu(
    returns,
    Sigma,
    halflife_days=EWMA_HALFLIFE_DAYS,
    tau=BL_TAU,
    delta=BL_DELTA,
    view_uncertainty=BL_VIEW_UNCERTAINTY,
    use_relative_views=True,
):
    """
    Black-Litterman posterior expected returns.

    Combines:
      - absolute EWMA views (P_abs = I) shrunk hard toward equilibrium, and
      - relative residual-mean-reversion views between correlated peers
        (P_rel rows = e_i - β e_j), with Ω inflated when the link is unstable.

    The relative block is the architectural upgrade suggested by the 2016
    correlation-divergence study, mapped into the existing BL + long-only QP
    (tilts, not market-neutral pairs).
    """
    syms = list(returns.columns)
    n = len(syms)
    Sig = Sigma.values

    vol = np.sqrt(np.clip(np.diag(Sig), 1e-12, None))
    w_mkt = 1.0 / vol
    w_mkt = w_mkt / w_mkt.sum()
    w_mkt_scaled = w_mkt * (delta / tau)

    # Absolute views (shrunk).
    q_abs = exp_weighted_mean_returns(returns, halflife_days).values
    P_abs = np.eye(n)
    om_abs = np.clip(np.diag(P_abs @ (tau * Sig) @ P_abs.T), 1e-12, None) * view_uncertainty

    if use_relative_views:
        P_rel, q_rel, om_rel = _relative_pair_views(returns, Sigma)
    else:
        P_rel = np.zeros((0, n))
        q_rel = np.zeros(0)
        om_rel = np.zeros(0)

    if len(q_rel) > 0:
        P = np.vstack([P_abs, P_rel])
        q = np.concatenate([q_abs, q_rel])
        Omega = np.diag(np.concatenate([om_abs, om_rel]))
    else:
        P, q, Omega = P_abs, q_abs, np.diag(om_abs)

    try:
        pi_bl = black_litterman_posterior(Sig, w_mkt_scaled, P, q, Omega, tau=tau)
    except np.linalg.LinAlgError:
        pi_bl = q_abs
    return pd.Series(np.asarray(pi_bl).reshape(-1), index=syms)


# -----------------------
# Optimizer
# -----------------------
def _apply_vol_target(w, Sigma, target_vol, max_invest, w_max):
    """Scale gross exposure toward `target_vol` without breaching caps."""
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
    costs=None,
    max_invest_fraction=MAX_INVEST,
    lambda_risk=LAMBDA_RISK,
    target_vol=TARGET_VOL,
):
    """
    Mean-variance QP with asset-specific turnover costs:

        max  mu'w - λ ||L'w||^2 - c'|w - w_prev|

    The vector cost ``c`` (from ``estimate_trading_costs``) creates a no-trade
    region: a weight only moves when its expected benefit exceeds its marginal
    trading cost — the Boyd / 2016-report "do nothing" zone, without a
    hand-tuned hard turnover cap.
    """
    n = len(mu)
    w = cp.Variable(n)
    L = np.linalg.cholesky(Sigma.values + 1e-8 * np.eye(n))

    if w_prev is None:
        w_prev = np.zeros(n)
    else:
        w_prev = np.asarray(w_prev, dtype=float).reshape(-1)

    if costs is None:
        costs = np.full(n, BASE_COST_BPS / 1e4)
    else:
        costs = np.asarray(costs, dtype=float).reshape(-1)

    obj = (
        mu.values @ w
        - lambda_risk * cp.sum_squares(L.T @ w)
        - costs @ cp.abs(w - w_prev)
    )
    constraints = [
        cp.sum(w) <= max_invest_fraction,
        w >= 0,
        w <= W_MAX,
    ]

    prob = cp.Problem(cp.Maximize(obj), constraints)
    prob.solve(solver=cp.OSQP, verbose=False, max_iter=10000)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"Optimizer warning: {prob.status}")

    w_opt = pd.Series(
        np.clip(w.value if w.value is not None else np.zeros(n), 0, 1),
        index=mu.index,
    )
    total_alloc = w_opt.sum()
    if total_alloc > max_invest_fraction:
        w_opt = w_opt * (max_invest_fraction / total_alloc)

    if target_vol and target_vol > 0:
        w_opt = _apply_vol_target(w_opt, Sigma, target_vol, max_invest_fraction, W_MAX)

    return w_opt


def _apply_trade_costs(equity, w_new, w_old, costs):
    """Deduct asset-specific costs from equity: equity *= 1 - sum c_i |dw_i|."""
    dw = (w_new - w_old).abs()
    c = costs.reindex(dw.index).fillna(BASE_COST_BPS / 1e4)
    drag = float((c * dw).sum())
    if drag > 0:
        equity *= (1.0 - drag)
    return equity


# Run Backtest
def walk_forward_backtest(px):
    rets = px.pct_change().dropna()
    dates = rets.index

    w_prev = None
    current_w = pd.Series(0.0, index=px.columns)
    held_w = pd.Series(0.0, index=px.columns)
    equity = 1.0
    equity_curve = []
    weights_record = {}
    last_rebal = -10 ** 9
    costs = pd.Series(BASE_COST_BPS / 1e4, index=px.columns)

    for t_idx, today in enumerate(dates):
        if t_idx >= RETURN_LOOKBACK_DAYS and (t_idx - last_rebal) >= REBAL_EVERY_DAYS:
            window = rets.iloc[t_idx - RETURN_LOOKBACK_DAYS:t_idx]
            assert window.index[-1] < today, "lookahead: window must end before the trade day"
            Sigma = estimate_cov(window)
            mu = black_litterman_mu(window, Sigma)
            costs = estimate_trading_costs(window)
            current_w = solve_portfolio(
                mu, Sigma, w_prev, costs=costs.values, lambda_risk=LAMBDA_RISK,
            )
            weights_record[today] = current_w
            w_prev = current_w.values.copy()
            last_rebal = t_idx

        equity = _apply_trade_costs(equity, current_w, held_w, costs)

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
    costs = pd.Series(BASE_COST_BPS / 1e4, index=pool_px.columns)
    universe = None
    last_uni_refresh = -10 ** 9
    last_rebal = -10 ** 9
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
                last_rebal = -10 ** 9  # force a rebalance into the new universe

            if (t_idx - last_rebal) >= REBAL_EVERY_DAYS:
                window = rets.iloc[t_idx - RETURN_LOOKBACK_DAYS:t_idx][universe]
                Sigma = estimate_cov(window)
                mu = black_litterman_mu(window, Sigma)
                costs_u = estimate_trading_costs(window)
                cur = solve_portfolio(
                    mu, Sigma, w_prev=w[universe].values,
                    costs=costs_u.values, lambda_risk=lambda_risk,
                )

                w = pd.Series(0.0, index=pool_px.columns)
                w[universe] = cur.values
                costs = pd.Series(BASE_COST_BPS / 1e4, index=pool_px.columns)
                costs[universe] = costs_u.values
                weights_record[today] = w.copy()
                last_rebal = t_idx

        equity = _apply_trade_costs(equity, w, held_w, costs)

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
        print(f"Backtest (asset-specific costs, daily rebal): Return {ann_ret:.2%}, "
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

        Sigma = estimate_cov(window)
        mu = black_litterman_mu(window, Sigma)
        costs = estimate_trading_costs(window)
        w = solve_portfolio(
            mu, Sigma, w_prev=current_w.values,
            costs=costs.values, lambda_risk=LAMBDA_RISK,
        )
        print("Target weights:\n", w.round(4))
        print("Per-asset one-way costs (bps):\n", (costs * 1e4).round(2))
        rebalance_alpaca_to_weights(w, notional=equity)

if __name__ == "__main__":
    main()
