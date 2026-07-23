import os
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Total-return (splits + dividends) adjustment for fair, leak-free prices.
try:
    from alpaca.data.enums import Adjustment
    _ADJ_ALL = Adjustment.ALL
except Exception:
    _ADJ_ALL = "all"

# Broad candidate pool spanning all major asset classes and sectors.
# The analyzer selects a diversified subset from these automatically.
#
# SURVIVORSHIP-BIAS CAVEAT: this is a hand-picked list of tickers that are still
# liquid and listed *today*. It therefore excludes names that were delisted,
# merged, or went to zero over the backtest window. Backtests on this pool will
# be biased upward relative to a true point-in-time universe, because we only
# ever trade companies that we already know survived. Universe *selection* is
# still point-in-time (see `end_date`), but pool *membership* is not. Treat
# absolute returns as optimistic; the risk-adjusted comparisons are more robust.
CANDIDATE_POOL = [
    # Broad market
    "SPY", "QQQ", "IWM", "DIA", "VTI",
    # Sector ETFs (all 11 GICS sectors)
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
    # International
    "EFA", "EEM", "FXI",
    # Fixed income
    "TLT", "IEF", "LQD", "HYG", "BND",
    # Commodities
    "GLD", "SLV", "USO", "PDBC",
    # Large-cap liquid equities (spread across sectors)
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    "JPM", "BAC", "GS",
    "UNH", "JNJ", "PFE",
    "XOM", "CVX",
    "CAT", "GE", "BA", "ITA",
    "MU", "TSM", "AVGO",
    "PG", "KO", "MCD",
]


def _fetch_candidates(end_date, lookback_days):
    key = os.environ["APCA_API_KEY_ID"]
    secret = os.environ["APCA_API_SECRET_KEY"]
    client = StockHistoricalDataClient(key, secret)

    end = pd.to_datetime(end_date) if end_date else pd.Timestamp(dt.date.today())
    # Fetch 1.5x calendar days to ensure enough trading days after dropping weekends/holidays
    start = end - pd.Timedelta(days=int(lookback_days * 1.5))

    request = StockBarsRequest(
        symbol_or_symbols=CANDIDATE_POOL,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        adjustment=_ADJ_ALL,
    )
    bars = client.get_stock_bars(request).df
    px = bars["close"].unstack(level=0).sort_index()
    return px


def _compute_pca_loadings(returns, n_components):
    """
    Fit PCA on standardized returns.
    Returns (loadings, pca) where loadings has shape (n_assets, n_components).
    Each row is an asset's coordinates in factor space.
    """
    R_scaled = StandardScaler().fit_transform(returns.values)

    if n_components is None:
        pca_full = PCA().fit(R_scaled)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumvar, 0.80) + 1)
        n_components = max(2, min(n_components, returns.shape[1] - 1))

    pca = PCA(n_components=n_components).fit(R_scaled)
    # components_ is (n_components, n_assets); transpose to get one row per asset
    return pca.components_.T, pca


def _cluster_assets(loadings, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(loadings)
    return labels, kmeans


def _score_assets(returns, criterion, loadings, labels, kmeans):
    if criterion == "sharpe":
        ann_ret = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)
        return ann_ret / (ann_vol + 1e-9)

    if criterion == "calmar":
        scores = {}
        for sym in returns.columns:
            r = returns[sym]
            ann_ret = r.mean() * 252
            curve = (1 + r).cumprod()
            max_dd = ((curve - curve.cummax()) / curve.cummax()).min()
            scores[sym] = ann_ret / (abs(max_dd) + 1e-9)
        return pd.Series(scores)

    if criterion == "centroid":
        # Prefer the asset closest to its cluster centroid (most "typical" member)
        centroids = kmeans.cluster_centers_
        scores = {}
        for i, sym in enumerate(returns.columns):
            dist = np.linalg.norm(loadings[i] - centroids[labels[i]])
            scores[sym] = -dist
        return pd.Series(scores)

    raise ValueError(f"Unknown criterion {criterion!r}. Choose 'sharpe', 'calmar', or 'centroid'.")


def _uniform_counts(labels, target_size):
    """One slot per cluster (used when n_clusters == target_size)."""
    return {c: 1 for c in sorted(set(labels))}


def _adaptive_counts(labels, scores, symbols, target_size, temperature):
    """
    Distribute `target_size` slots across clusters proportional to each cluster's
    average member score, so stronger-performing clusters get more representatives
    (and weak clusters may get zero). Uses softmax weighting + largest-remainder
    rounding, capped at each cluster's membership.
    """
    clusters = sorted(set(labels))
    sizes = {c: sum(1 for l in labels if l == c) for c in clusters}
    strength = np.array([
        scores[[symbols[i] for i, l in enumerate(labels) if l == c]].mean()
        for c in clusters
    ])

    # Scale-normalize then softmax so allocation is comparable across runs.
    s = (strength - strength.mean()) / (strength.std() + 1e-9) / max(temperature, 1e-6)
    weights = np.exp(s - s.max())
    weights /= weights.sum()

    raw = weights * target_size
    counts = {c: min(int(np.floor(raw[i])), sizes[c]) for i, c in enumerate(clusters)}

    # Largest-remainder pass to reach exactly target_size, respecting cluster sizes.
    remainders = sorted(
        range(len(clusters)),
        key=lambda i: raw[i] - np.floor(raw[i]),
        reverse=True,
    )
    short = target_size - sum(counts.values())
    guard = 0
    while short > 0 and guard < 10000:
        for i in remainders:
            c = clusters[i]
            if counts[c] < sizes[c]:
                counts[c] += 1
                short -= 1
                if short == 0:
                    break
        guard += 1
    return counts


def _pick_top_per_cluster(labels, scores, symbols, counts):
    """Take the top-`counts[c]` assets (by score) from each cluster."""
    selected = []
    for c in sorted(counts):
        members = [sym for i, sym in enumerate(symbols) if labels[i] == c]
        ranked = scores[members].sort_values(ascending=False)
        selected.extend(ranked.index[: counts[c]].tolist())
    return selected


def build_universe(
    lookback_days=504,
    target_size=10,
    n_components=None,
    criterion="sharpe",
    allocation="uniform",
    n_clusters=None,
    temperature=1.0,
    end_date=None,
    verbose=False,
):
    """
    Select `target_size` approximately orthogonal assets from CANDIDATE_POOL.

    Algorithm:
      1. PCA on standardized daily returns → each asset gets coordinates in factor space
      2. K-means clusters assets by factor exposure
      3. Allocate slots to clusters, then pick the top-scoring asset(s) in each

    Parameters
    ----------
    lookback_days : trading days of history to use (default ~2 years)
    target_size   : number of assets to return
    n_components  : PCA components to keep; None = auto (explain >= 80% variance)
    criterion     : selection rule within a cluster: "sharpe" (default), "calmar", "centroid"
    allocation    : "uniform" = one asset per cluster (k = target_size clusters);
                    "adaptive" = fewer clusters, slots distributed by cluster strength
                    so stronger clusters contribute more assets.
    n_clusters    : number of clusters; None auto-picks (target_size for uniform,
                    ~0.6*target_size for adaptive).
    temperature   : adaptive softmax temperature (lower = more concentrated tilt).
    end_date      : ISO date string (YYYY-MM-DD); set to backtest start to avoid lookahead bias
    verbose       : print cluster report to stdout
    """
    px = _fetch_candidates(end_date, lookback_days)

    # Drop assets with > 20% missing observations, then forward-fill remaining gaps
    px = px.dropna(thresh=int(lookback_days * 0.8), axis=1).ffill().dropna()
    px = px.iloc[-lookback_days:]

    returns = px.pct_change().dropna()
    return select_universe(
        returns,
        target_size=target_size,
        n_components=n_components,
        criterion=criterion,
        allocation=allocation,
        n_clusters=n_clusters,
        temperature=temperature,
        verbose=verbose,
    )


def select_universe(
    returns,
    target_size=10,
    n_components=None,
    criterion="sharpe",
    allocation="uniform",
    n_clusters=None,
    temperature=1.0,
    verbose=False,
):
    """
    Select a universe from an in-memory returns DataFrame (no data fetching).

    This is the pure selection core shared by build_universe and by rolling-universe
    backtests, which call it repeatedly on trailing windows. Columns containing any
    NaN over `returns` are dropped first.

    See build_universe for parameter meanings.
    """
    returns = returns.dropna(axis=1)
    symbols = list(returns.columns)
    N = len(symbols)

    if N < target_size:
        raise RuntimeError(
            f"Only {N} assets available; need >= {target_size}. "
            "Provide more history or reduce target_size."
        )

    if n_clusters is None:
        n_clusters = target_size if allocation == "uniform" else max(3, round(target_size * 0.6))
    n_clusters = max(1, min(n_clusters, N))

    loadings, pca = _compute_pca_loadings(returns, n_components)
    labels, kmeans = _cluster_assets(loadings, n_clusters)
    scores = _score_assets(returns, criterion, loadings, labels, kmeans)

    if allocation == "adaptive":
        counts = _adaptive_counts(labels, scores, symbols, target_size, temperature)
    elif allocation == "uniform":
        counts = _uniform_counts(labels, target_size)
    else:
        raise ValueError(f"Unknown allocation {allocation!r}. Choose 'uniform' or 'adaptive'.")

    universe = _pick_top_per_cluster(labels, scores, symbols, counts)

    if verbose:
        var_explained = pca.explained_variance_ratio_.sum()
        print(f"PCA: {pca.n_components_} components -> {var_explained:.1%} variance explained")
        print(f"Clustering {N} candidates into {n_clusters} groups "
              f"(criterion: {criterion}, allocation: {allocation})\n")
        for cid in sorted(set(labels)):
            members = [sym for i, sym in enumerate(symbols) if labels[i] == cid]
            picks = set(s for s in members if s in universe)
            member_str = ", ".join(f"[{s}]" if s in picks else s for s in sorted(members))
            print(f"  Cluster {cid:>2} (x{counts.get(cid, 0)}): {member_str}")
        print(f"\nSelected universe ({len(universe)}): {universe}")

    return universe


def analyze_and_print(lookback_days=504, target_size=10):
    build_universe(lookback_days=lookback_days, target_size=target_size, verbose=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Survey CANDIDATE_POOL and select a diversified trading universe via PCA + clustering"
    )
    parser.add_argument("--lookback", type=int, default=504, help="Days of history to analyze (default: 504 ≈ 2yr)")
    parser.add_argument("--size", type=int, default=10, help="Universe size (default: 10)")
    parser.add_argument(
        "--criterion", default="sharpe", choices=["sharpe", "calmar", "centroid"],
        help="How to pick the representative from each cluster (default: sharpe)"
    )
    parser.add_argument(
        "--allocation", default="uniform", choices=["uniform", "adaptive"],
        help="uniform = 1 asset/cluster; adaptive = tilt slots toward stronger clusters (default: uniform)"
    )
    parser.add_argument("--clusters", type=int, default=None, help="Number of clusters (default: auto)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Adaptive softmax temperature (default: 1.0)")
    parser.add_argument(
        "--end-date", default=None,
        help="ISO date YYYY-MM-DD; omit for today. Set to backtest start date to avoid lookahead bias."
    )
    args = parser.parse_args()

    build_universe(
        lookback_days=args.lookback,
        target_size=args.size,
        criterion=args.criterion,
        allocation=args.allocation,
        n_clusters=args.clusters,
        temperature=args.temperature,
        end_date=args.end_date,
        verbose=True,
    )
