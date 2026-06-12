# E6 — CVaR Trader: Fixed vs Rolling Universe

**Window:** 2025-06-10 → 2026-06-10 (252 trading days)  
**Universe selected as of:** 2023-12-24 (no lookahead)

The universe is re-surveyed every ~quarter (63 trading days) using only past data, so the portfolio can rotate into newly-surging leaders. Three modes:

- **fixed** — one universe chosen at the start and held;
- **rolling / uniform** — re-selected each quarter, one asset per cluster;
- **rolling / adaptive** — re-selected each quarter *and* the per-cluster slot allocation re-tilts toward the strongest-performing clusters.

> **Caveat:** rolling selection picks *recent* winners, which tend to mean-revert; it also adds turnover (not charged here). Compare the table below — a stable, diversified universe can beat quarterly winner-chasing in a trending market.

### Rolling adaptive universe over time

6 re-selections:

- 2024-12-27: `['XLF', 'GS', 'JPM', 'XLC', 'SPY', 'TSM', 'VTI', 'XLU']`
- 2025-04-01: `['XLF', 'JPM', 'GS', 'BAC', 'GLD', 'XOM', 'XLU', 'KO']`
- 2025-07-02: `['GLD', 'XLU', 'SLV', 'TSM', 'FXI', 'JPM', 'GS', 'XLC']`
- 2025-10-01: `['BND', 'ITA', 'GS', 'XLU', 'GLD', 'SLV', 'XLC', 'GOOGL']`
- 2025-12-31: `['ITA', 'GE', 'MU', 'GLD', 'SLV', 'GS', 'XOM', 'BND']`
- 2026-04-02: `['GLD', 'JNJ', 'KO', 'USO', 'XOM', 'CVX', 'PDBC', 'MU']`

**Fixed universe:** `['GLD', 'AVGO', 'UNH', 'LQD', 'JPM', 'XOM', 'MCD', 'GOOGL']`

## Results

| Strategy | Ann. Return | Ann. Vol | Sharpe | Max Drawdown |
|---|---|---|---|---|
| CVaR fixed | 22.73% | 9.43% | 2.41 | -5.76% |
| CVaR rolling (uniform) | 17.15% | 11.09% | 1.55 | -7.90% |
| CVaR rolling (adaptive) | 8.77% | 12.62% | 0.69 | -12.16% |
| S&P 500 (SPY) | 20.29% | 12.23% | 1.66 | -9.13% |
| Dow Jones (DIA) | 16.44% | 12.40% | 1.33 | -10.06% |

![results](e6_cvar_rolling.png)
