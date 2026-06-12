# E7 — Mean-Variance Trader: Fixed vs Rolling Universe

**Window:** 2025-06-10 → 2026-06-10 (252 trading days)  
**Universe selected as of:** 2023-12-24 (no lookahead)

Mean-variance trader with a fixed universe vs a rolling universe re-surveyed every ~quarter with adaptive cluster weighting (slots tilt toward recently strong clusters). Rolling variant uses static risk-aversion for simplicity.

> **Caveat:** chasing recent winners quarterly adds turnover and timing risk; see whether it actually improves risk-adjusted returns in the table below.

### Rolling adaptive universe over time

9 re-selections:

- 2024-05-21: `['NVDA', 'PG', 'TSM', 'GLD', 'XLF', 'XOM', 'XLE', 'XLV', 'GE']`
- 2024-08-21: `['KO', 'XLP', 'XLU', 'GLD', 'XLC', 'META', 'TSM', 'SPY', 'XLF']`
- 2024-11-19: `['XLU', 'XLP', 'XLC', 'GLD', 'META', 'AMZN', 'XLF', 'GS', 'XOM']`
- 2025-02-24: `['HYG', 'XLF', 'JPM', 'GS', 'BAC', 'GLD', 'XLC', 'XOM', 'TSM']`
- 2025-05-23: `['JPM', 'GE', 'XLF', 'XOM', 'GLD', 'FXI', 'XLC', 'META', 'TSM']`
- 2025-08-25: `['GLD', 'ITA', 'GS', 'XLU', 'KO', 'XLC', 'SPY', 'META', 'VTI']`
- 2025-11-21: `['BND', 'GE', 'ITA', 'GS', 'XLU', 'GLD', 'GOOGL', 'SLV', 'XLC']`
- 2026-02-25: `['GLD', 'CAT', 'SLV', 'ITA', 'XLK', 'BND', 'XOM', 'CVX', 'PDBC']`
- 2026-05-27: `['MU', 'CAT', 'EEM', 'TSM', 'GOOGL', 'GLD', 'SLV', 'USO', 'BND']`

**Fixed universe:** `['CAT', 'AVGO', 'XOM', 'GLD', 'EEM', 'LQD', 'GOOGL', 'UNH', 'MCD']`

## Results

| Strategy | Ann. Return | Ann. Vol | Sharpe | Max Drawdown |
|---|---|---|---|---|
| MV fixed | 67.39% | 13.83% | 4.87 | -4.85% |
| MV rolling (adaptive) | 43.19% | 26.43% | 1.63 | -19.15% |
| S&P 500 (SPY) | 20.29% | 12.23% | 1.66 | -9.13% |
| Dow Jones (DIA) | 16.44% | 12.40% | 1.33 | -10.06% |

![results](e7_mv_rolling.png)
