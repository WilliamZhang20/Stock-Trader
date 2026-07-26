# E6 — CVaR Trader: Fixed vs Rolling Universe

**Window:** 2025-07-24 → 2026-07-24 (252 trading days)  
**Universe selected as of:** 2022-06-17 (no lookahead)

The universe is re-surveyed every ~quarter (63 trading days) using only past data, so the portfolio can rotate into newly-surging leaders. Three modes:

- **fixed** — one universe chosen at the start and held;
- **rolling / uniform** — re-selected each quarter, one asset per cluster;
- **rolling / adaptive** — re-selected each quarter *and* the per-cluster slot allocation re-tilts toward the strongest-performing clusters.

> **Caveat:** rolling selection picks *recent* winners, which tend to mean-revert; it also adds turnover (not charged here). Compare the table below — a stable, diversified universe can beat quarterly winner-chasing in a trending market.

### Rolling adaptive universe over time

13 re-selections:

- 2023-06-22: `['NVDA', 'AVGO', 'XLK', 'AAPL', 'XOM', 'GE', 'BA', 'MCD']`
- 2023-09-21: `['GE', 'NVDA', 'AVGO', 'META', 'CAT', 'GLD', 'XOM', 'XLE']`
- 2023-12-20: `['MCD', 'GE', 'BA', 'AVGO', 'NVDA', 'XLK', 'QQQ', 'META']`
- 2024-03-22: `['NVDA', 'AVGO', 'XLK', 'MU', 'TSM', 'GE', 'XOM', 'PG']`
- 2024-06-24: `['JPM', 'XLF', 'NVDA', 'META', 'QQQ', 'XLK', 'XLC', 'GE']`
- 2024-09-23: `['HYG', 'KO', 'XLP', 'GE', 'NVDA', 'SPY', 'META', 'VTI']`
- 2024-12-20: `['NVDA', 'AVGO', 'QQQ', 'XLC', 'META', 'AMZN', 'GOOGL', 'XLF']`
- 2025-03-26: `['NVDA', 'SPY', 'XLC', 'META', 'AMZN', 'GE', 'GLD', 'HYG']`
- 2025-06-26: `['GE', 'NVDA', 'XLU', 'GLD', 'XLC', 'META', 'SLV', 'MSFT']`
- 2025-09-25: `['GE', 'HYG', 'ITA', 'GLD', 'SLV', 'BND', 'LQD', 'CVX']`
- 2025-12-24: `['HYG', 'GS', 'ITA', 'GLD', 'SLV', 'BND', 'LQD', 'XOM']`
- 2026-03-27: `['XLU', 'MU', 'ITA', 'GE', 'PDBC', 'CVX', 'XOM', 'XLE']`
- 2026-06-29: `['BND', 'MU', 'JNJ', 'GE', 'ITA', 'GLD', 'SLV', 'PDBC']`

**Fixed universe:** `['UNH', 'PDBC', 'TSLA', 'KO', 'CAT', 'LQD', 'EEM', 'SLV']`

## Results

| Strategy | Ann. Return | Ann. Vol | Sharpe (excess) | Max Drawdown |
|---|---|---|---|---|
| CVaR fixed | 38.43% | 10.82% | 3.18 | -5.06% |
| CVaR rolling (uniform) | 22.66% | 7.91% | 2.36 | -5.46% |
| CVaR rolling (adaptive) | 18.50% | 9.11% | 1.59 | -7.74% |
| S&P 500 (SPY) | 17.77% | 12.71% | 1.08 | -8.88% |
| Dow Jones (DIA) | 17.77% | 12.26% | 1.12 | -9.76% |
| 60/40 (SPY/IEF) | 11.52% | 8.25% | 0.91 | -6.00% |

![results](e6_cvar_rolling.png)
