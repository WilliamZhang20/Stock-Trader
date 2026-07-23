# E7 — Mean-Variance Trader: Fixed vs Rolling Universe

**Window:** 2025-07-22 → 2026-07-22 (252 trading days)  
**Universe selected as of:** 2022-06-14 (no lookahead)

Mean-variance trader with a fixed universe vs a rolling universe re-surveyed every ~quarter with adaptive cluster weighting (slots tilt toward recently strong clusters). Rolling variant uses static risk-aversion for simplicity.

> **Caveat:** chasing recent winners quarterly adds turnover and timing risk; see whether it actually improves risk-adjusted returns in the table below.

### Rolling adaptive universe over time

15 re-selections:

- 2022-11-07: `['AAPL', 'GS', 'BA', 'GE', 'MCD', 'UNH', 'XLV', 'XOM', 'CVX']`
- 2023-02-08: `['BA', 'GE', 'GS', 'JPM', 'ITA', 'XLF', 'MCD', 'NVDA', 'EFA']`
- 2023-05-10: `['NVDA', 'AAPL', 'GE', 'BA', 'EFA', 'ITA', 'MCD', 'PG', 'LQD']`
- 2023-08-10: `['GE', 'NVDA', 'BA', 'AVGO', 'JPM', 'META', 'MCD', 'GLD', 'XLV']`
- 2023-11-08: `['NVDA', 'AVGO', 'META', 'XLK', 'MSFT', 'PG', 'UNH', 'GE', 'BA']`
- 2024-02-09: `['NVDA', 'AVGO', 'META', 'XLK', 'MSFT', 'GE', 'JPM', 'XOM', 'MCD']`
- 2024-05-10: `['HYG', 'PG', 'GE', 'META', 'NVDA', 'AVGO', 'XLC', 'ITA', 'BA']`
- 2024-08-12: `['GE', 'NVDA', 'META', 'AVGO', 'XLC', 'QQQ', 'JPM', 'HYG', 'XLV']`
- 2024-11-08: `['XLU', 'XLF', 'GE', 'NVDA', 'XLC', 'SPY', 'VTI', 'META', 'HYG']`
- 2025-02-12: `['SPY', 'NVDA', 'VTI', 'XLC', 'META', 'AMZN', 'GOOGL', 'GE', 'HYG']`
- 2025-05-14: `['GE', 'HYG', 'JPM', 'XLF', 'GLD', 'EFA', 'BND', 'LQD', 'XOM']`
- 2025-08-14: `['GE', 'XLU', 'HYG', 'GLD', 'FXI', 'SLV', 'BND', 'LQD', 'XOM']`
- 2025-11-12: `['ITA', 'GE', 'AVGO', 'TSM', 'HYG', 'GS', 'GLD', 'BND', 'IEF']`
- 2026-02-13: `['MU', 'CAT', 'GLD', 'JNJ', 'BND', 'CVX', 'ITA', 'GE', 'BA']`
- 2026-05-15: `['MU', 'TSM', 'GOOGL', 'CAT', 'JNJ', 'BND', 'PDBC', 'USO', 'XOM']`

**Fixed universe:** `['AVGO', 'LQD', 'CAT', 'PDBC', 'KO', 'SLV', 'EEM', 'TSLA', 'UNH']`

## Results

| Strategy | Ann. Return | Ann. Vol | Sharpe (excess) | Max Drawdown |
|---|---|---|---|---|
| MV fixed | 12.28% | 13.85% | 0.60 | -10.19% |
| MV rolling (adaptive) | 20.68% | 15.21% | 1.10 | -8.34% |
| S&P 500 (SPY) | 20.17% | 12.66% | 1.28 | -8.88% |
| Dow Jones (DIA) | 18.96% | 12.28% | 1.22 | -9.76% |
| 60/40 (SPY/IEF) | 12.68% | 8.20% | 1.06 | -6.00% |

![results](e7_mv_rolling.png)
