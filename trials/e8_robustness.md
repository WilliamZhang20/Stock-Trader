# E8 - Robustness: Costs, Out-of-Sample, Deflated Sharpe

**Window:** 2025-07-24 → 2026-07-24 (252 trading days)  
**Universe selected as of:** 2022-06-17 (no lookahead)

Directly addresses the four reasons a high backtested return draws skepticism: overfitting, transaction costs, data leakage, and benchmark selection.

### 1. Transaction costs (gross vs net, last 252d)
Net charges 10 bps per unit of L1 turnover on every rebalance; gross charges nothing. The gap is the honest cost drag:

| Strategy | Ann. Return | Ann. Vol | Sharpe (excess) | Max Drawdown |
|---|---|---|---|---|
| CVaR gross (0 bps) | 41.99% | 10.89% | 3.49 | -5.05% |
| CVaR net (10 bps) | 38.43% | 10.82% | 3.18 | -5.06% |
| Mean-Variance gross (0 bps) | 42.93% | 14.08% | 2.77 | -7.10% |
| Mean-Variance net (10 bps) | 42.31% | 14.08% | 2.72 | -7.21% |

### 2. Overfitting - out-of-sample consistency
The same frozen strategy scored on consecutive, non-overlapping ~1y sub-windows (net of costs). A single lucky year is easy; consistency across windows is not:

| Window | Ann. Return | Sharpe (excess) | Max Drawdown |
|---|---|---|---|
| CVaR W1 (2023-06-22->2024-06-21) | 9.49% | 0.71 | -7.14% |
| CVaR W2 (2024-06-24->2025-06-25) | -0.37% | -0.49 | -9.33% |
| CVaR W3 (2025-06-26->2026-06-26) | 39.70% | 3.38 | -5.06% |
| **CVaR mean** | **16.27%** | **1.20** | - |
| **CVaR worst** | **-0.37%** | **-0.49** | - |
| Mean-Variance W1 (2023-06-22->2024-06-21) | 4.21% | 0.02 | -5.25% |
| Mean-Variance W2 (2024-06-24->2025-06-25) | 0.00% | -0.35 | -12.60% |
| Mean-Variance W3 (2025-06-26->2026-06-26) | 49.88% | 3.29 | -7.21% |
| **Mean-Variance mean** | **18.03%** | **0.99** | - |
| **Mean-Variance worst** | **0.00%** | **-0.35** | - |

### 2b. Overfitting - deflated Sharpe (multiple testing)
Probability the excess Sharpe is real after ~8 configurations were tried on this data: **CVaR 0.87**, **Mean-Variance 0.76** (near 1.0 survives skepticism; near 0.5 is plausibly a fluke of selection).

### 3. Data leakage
Universe selection is point-in-time (`end_date=SELECT_END`); each trade uses only returns strictly before the trade day (now asserted in both backtests). Prices are split+dividend adjusted. Remaining bias: `CANDIDATE_POOL` is a survivors list (documented in `market_analyzer.py`), so absolute returns are optimistic.

### 4. Benchmark selection
Benchmarks use dividend-adjusted total-return prices, a 4% risk-free rate for excess Sharpe, and add a 60/40 (SPY/IEF) comparator since these strategies often hold cash and run below-market volatility.

**CVaR universe:** `['UNH', 'PDBC', 'TSLA', 'KO', 'CAT', 'LQD', 'EEM', 'SLV']`

**Mean-Variance universe:** `['TSLA', 'LQD', 'CAT', 'PDBC', 'KO', 'SLV', 'EEM', 'UNH', 'XLI']`

## Results

| Strategy | Ann. Return | Ann. Vol | Sharpe (excess) | Max Drawdown |
|---|---|---|---|---|
| CVaR (net) | 38.43% | 10.82% | 3.18 | -5.06% |
| Mean-Variance (net) | 42.31% | 14.08% | 2.72 | -7.21% |
| S&P 500 (SPY) | 17.77% | 12.71% | 1.08 | -8.88% |
| Dow Jones (DIA) | 17.77% | 12.26% | 1.12 | -9.76% |
| 60/40 (SPY/IEF) | 11.52% | 8.25% | 0.91 | -6.00% |

![results](e8_robustness.png)
