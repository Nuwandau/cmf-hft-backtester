# HFT Market-Making Backtest Report

## Objective

Develop an event-driven LOB replay backtester and compare Avellaneda-Stoikov market-making with mid-price and finite-state microprice references.

## Current Experiment Setup

- Historical replay uses L1 top-of-book features extracted from 25-level L2 snapshots.
- Execution uses crossing rules: buy fills when future best ask crosses our bid, sell fills when future best bid crosses our ask.
- Orders placed at timestamp `t` are eligible only from later snapshots.
- Final historical runs use `visible_size` partial-fill approximation, zero fees, no latency, and no queue-position model.
- The final AS configs use `quote_refresh_seconds = 0.25` and `microprice.max_mid_move_ticks = 10.0`, both selected or checked on validation.

## Strategy Models

```text
r_t = reference_t - q_t * gamma * sigma_t^2 * tau
spread_t = gamma * sigma_t^2 * tau + 2/gamma * log(1 + gamma/k)
bid_t = r_t - spread_t / 2
ask_t = r_t + spread_t / 2
```

`reference_t` is the mid-price in the baseline strategy and the finite-state microprice in the enhanced strategy. Portfolio inventory is stored in raw amount units; the AS inventory term uses `portfolio_inventory / inventory_risk_unit`.

## Data Audit

- `rows`: `1036690`
- `timestamp_order_violations`: `0`
- `duplicate_timestamps`: `0`
- `median_spread_ticks`: `1`
- `p99_spread_ticks`: `14`
- `max_spread_ticks`: `692`
- `mean_imbalance`: `0.500672`

## Data Regime By Date

| date | split | rows | median_spread_ticks | p99_spread_ticks | fraction_one_tick_spread | fraction_spread_gt_10_ticks | mean_imbalance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2024-08-01 | train | 172786 | 1 | 6 | 0.967451 | 0.003345 | 0.499154 |
| 2024-08-02 | train | 172791 | 1 | 7 | 0.950703 | 0.004323 | 0.503653 |
| 2024-08-03 | train | 172779 | 1 | 7 | 0.952679 | 0.00382 | 0.498044 |
| 2024-08-04 | validation | 172790 | 1 | 7 | 0.941571 | 0.004375 | 0.506782 |
| 2024-08-05 | test | 172758 | 1 | 28 | 0.588528 | 0.086109 | 0.495597 |
| 2024-08-06 | test | 172786 | 1 | 10 | 0.83766 | 0.009057 | 0.500801 |

Full table: `reports/tables/data_audit_by_date.csv`

## Final Historical Performance

Final performance is out-of-sample on the test split. PnL is gross mark-to-market PnL under crossing execution.

| strategy | final_pnl | max_drawdown | turnover | fill_count | final_inventory | max_abs_inventory | avg_quoted_spread_ticks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| avellaneda_stoikov_mid | -393.877 | 395.11 | 1,188,876.59 | 17466 | 3,620 | 38,599 | 106.525 |
| avellaneda_stoikov_microprice | -392.971 | 394.084 | 1,183,641.49 | 17401 | -1,279 | 38,453 | 106.528 |

Full table: `reports/tables/final_performance.csv`

## Validation Grid: AS Parameters

Grid search covers `gamma`, `k`, `tau`, inventory limit, and quote refresh. The table shows the top validation configurations; the full CSV keeps every run.

| gamma | k | tau_seconds | max_inventory | quote_refresh_seconds | score | final_pnl |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0001 | 0.025 | 180 | 100,000 | 0.1 | -80.083 | -72.694 |
| 0.0001 | 0.025 | 180 | 100,000 | 0.25 | -80.083 | -72.694 |
| 0.0001 | 0.025 | 180 | 50,000 | 0.1 | -80.145 | -72.751 |
| 0.0001 | 0.025 | 180 | 50,000 | 0.25 | -80.145 | -72.751 |
| 0.0001 | 0.025 | 60 | 100,000 | 0.1 | -92.975 | -84.056 |
| 0.0001 | 0.025 | 60 | 100,000 | 0.25 | -92.975 | -84.056 |
| 0.0001 | 0.025 | 60 | 50,000 | 0.1 | -95.337 | -86.203 |
| 0.0001 | 0.025 | 60 | 50,000 | 0.25 | -95.337 | -86.203 |
| 0.0001 | 0.025 | 30 | 50,000 | 0.1 | -97.747 | -88.124 |
| 0.0001 | 0.025 | 30 | 50,000 | 0.25 | -97.747 | -88.124 |
| 0.0001 | 0.025 | 30 | 100,000 | 0.1 | -98.111 | -88.097 |
| 0.0001 | 0.025 | 30 | 100,000 | 0.25 | -98.111 | -88.097 |

Full table: `reports/tables/calibration_results.csv`

## Volatility Sensitivity: Mid-Price AS

This check varies the realized-volatility window, volatility floor, and a direct multiplier on sigma. It tests sensitivity to the `sigma_t` input in AS; the full CSV contains both validation and test rows.

| vol_window_seconds | vol_floor | sigma_multiplier | score | final_pnl | fill_count | avg_quoted_spread_ticks |
| --- | --- | --- | --- | --- | --- | --- |
| 60 | 0.05 | 2 | -31.778 | -28.858 | 1764 | 118.616 |
| 60 | 0.1 | 2 | -31.778 | -28.858 | 1764 | 118.616 |
| 60 | 0.2 | 2 | -31.778 | -28.858 | 1764 | 118.616 |
| 180 | 0.1 | 2 | -34.559 | -31.384 | 1867 | 118.579 |
| 180 | 0.2 | 2 | -34.559 | -31.384 | 1867 | 118.579 |
| 180 | 0.05 | 2 | -34.559 | -31.384 | 1867 | 118.579 |

Full table: `reports/tables/volatility_sensitivity.csv`

## Volatility Sensitivity: Microprice AS

| vol_window_seconds | vol_floor | sigma_multiplier | score | final_pnl | fill_count | avg_quoted_spread_ticks |
| --- | --- | --- | --- | --- | --- | --- |
| 600 | 0.05 | 2 | -33.33 | -30.26 | 1523 | 118.564 |
| 600 | 0.2 | 2 | -33.33 | -30.26 | 1523 | 118.564 |
| 600 | 0.1 | 2 | -33.33 | -30.26 | 1523 | 118.564 |
| 300 | 0.1 | 2 | -33.352 | -30.283 | 1571 | 118.578 |
| 300 | 0.2 | 2 | -33.352 | -30.283 | 1571 | 118.578 |
| 300 | 0.05 | 2 | -33.352 | -30.283 | 1571 | 118.578 |

Full table: `reports/tables/volatility_sensitivity.csv`

## Empirical k Diagnostic

`k` is estimated on train from hypothetical quote crossing probabilities. It is used to choose a plausible validation grid, not as an exact fill model.

| horizon_seconds | horizon_events | median_dt_seconds | k_fit | n_fit_points |
| --- | --- | --- | --- | --- |
| 1 | 2 | 0.515774 | 0.063749 | 11 |
| 2.5 | 5 | 0.515774 | 0.048838 | 10 |
| 5 | 10 | 0.515774 | 0.036017 | 5 |
| 10 | 19 | 0.515774 | 0.028554 | 4 |
| 30 | 58 | 0.515774 | 0.018176 | 2 |

Full table: `reports/tables/k_estimation_summary.csv`

## Microprice Move-Filter Sensitivity

| max_mid_move_ticks | split | score | final_pnl | fill_count | filtered_share | max_abs_adjustment_ticks |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | test | -432.38 | -392.971 | 17401 | 0.322675 | 1.6379 |
| 1 | test | -433.548 | -394.022 | 17469 | 0.590144 | 0.190493 |
| 2 | test | -434.777 | -395.138 | 17492 | 0.558775 | 0.290378 |
| 5 | test | -438.893 | -398.876 | 17458 | 0.46885 | 0.825889 |
| 10 | validation | -76.774 | -69.663 | 3841 | 0.322675 | 1.6379 |
| 1 | validation | -80.214 | -72.813 | 3844 | 0.590144 | 0.190493 |
| 5 | validation | -80.981 | -73.521 | 3853 | 0.46885 | 0.825889 |
| 2 | validation | -81.97 | -74.41 | 3849 | 0.558775 | 0.290378 |

Full table: `reports/tables/microprice_move_sensitivity.csv`

## Quote Refresh Sensitivity

| strategy | quote_refresh_seconds | split | score | final_pnl | fill_count | avg_abs_inventory |
| --- | --- | --- | --- | --- | --- | --- |
| avellaneda_stoikov_microprice | 0.25 | test | -432.38 | -392.971 | 17401 | 5,432.71 |
| avellaneda_stoikov_microprice | 0.1 | test | -432.429 | -393.015 | 17399 | 5,432.72 |
| avellaneda_stoikov_microprice | 0.5 | test | -527.579 | -479.513 | 22443 | 5,835.73 |
| avellaneda_stoikov_microprice | 5 | test | -528.934 | -480.775 | 23942 | 9,689.83 |
| avellaneda_stoikov_microprice | 1 | test | -635.152 | -577.355 | 27251 | 6,603.08 |
| avellaneda_stoikov_microprice | 2 | test | -660.179 | -600.092 | 28738 | 7,681.34 |
| avellaneda_stoikov_microprice | 0.1 | validation | -76.774 | -69.663 | 3841 | 8,624.75 |
| avellaneda_stoikov_microprice | 0.25 | validation | -76.774 | -69.663 | 3841 | 8,624.75 |
| avellaneda_stoikov_microprice | 0.5 | validation | -103.089 | -93.609 | 5205 | 8,769.27 |
| avellaneda_stoikov_microprice | 1 | validation | -123.77 | -112.445 | 6734 | 8,780.14 |
| avellaneda_stoikov_microprice | 2 | validation | -128.946 | -117.158 | 7661 | 9,778.86 |
| avellaneda_stoikov_microprice | 5 | validation | -132.428 | -120.331 | 7374 | 11,421.32 |
| avellaneda_stoikov_mid | 0.25 | test | -433.389 | -393.877 | 17466 | 5,385.46 |
| avellaneda_stoikov_mid | 0.1 | test | -433.438 | -393.922 | 17464 | 5,385.47 |
| avellaneda_stoikov_mid | 5 | test | -526.083 | -478.162 | 23988 | 9,606.25 |
| avellaneda_stoikov_mid | 0.5 | test | -540.95 | -491.648 | 22650 | 5,827.64 |
| avellaneda_stoikov_mid | 1 | test | -633.703 | -576.019 | 27279 | 6,617.62 |
| avellaneda_stoikov_mid | 2 | test | -661.604 | -601.376 | 28737 | 7,600.92 |
| avellaneda_stoikov_mid | 0.1 | validation | -80.083 | -72.694 | 3852 | 8,937.95 |
| avellaneda_stoikov_mid | 0.25 | validation | -80.083 | -72.694 | 3852 | 8,937.95 |
| avellaneda_stoikov_mid | 0.5 | validation | -103.205 | -93.71 | 5329 | 8,518.9 |
| avellaneda_stoikov_mid | 1 | validation | -128.555 | -116.784 | 6754 | 8,776.4 |
| avellaneda_stoikov_mid | 2 | validation | -134.105 | -121.86 | 7761 | 9,850.5 |
| avellaneda_stoikov_mid | 5 | validation | -135.804 | -123.39 | 7365 | 11,595.5 |

Full table: `reports/tables/quote_refresh_sensitivity.csv`

## Microprice Signal Diagnostics

A correct directional microprice should be negative at low bid imbalance and positive at high bid imbalance.

| imbalance_bucket | n | mean_imbalance | mean_adjustment_ticks | mean_local_next_mid_move_ticks | local_transition_share |
| --- | --- | --- | --- | --- | --- |
| 1 | 97347 | 0.034489 | -1.3452 | -0.975396 | 0.657575 |
| 2 | 50910 | 0.147895 | -0.819583 | -0.548218 | 0.690493 |
| 3 | 41238 | 0.248753 | -0.550527 | -0.406679 | 0.694578 |
| 4 | 35848 | 0.349159 | -0.3256 | -0.241457 | 0.690638 |
| 5 | 33660 | 0.449695 | -0.086869 | -0.078395 | 0.69085 |
| 6 | 33540 | 0.54999 | 0.087495 | 0.036747 | 0.690877 |
| 7 | 35739 | 0.651099 | 0.325268 | 0.230966 | 0.688715 |
| 8 | 41563 | 0.751298 | 0.55173 | 0.371674 | 0.689123 |
| 9 | 50974 | 0.852066 | 0.819818 | 0.576323 | 0.687959 |
| 10 | 97536 | 0.965293 | 1.3444 | 0.964235 | 0.653892 |

Full table: `reports/tables/microprice_signal_diagnostics.csv`

## Strategy Similarity Diagnostics

This explains why mid-price and microprice AS can remain close in PnL: the fitted microprice adjustment is still small relative to the AS quoted spread, even when rounded quotes differ frequently.

| same_both_quotes_share | median_abs_microprice_adjustment_ticks | p99_abs_microprice_adjustment_ticks | avg_quoted_spread_ticks | p99_adjustment_to_spread_ratio | final_pnl_diff_micro_minus_mid | fill_count_diff_micro_minus_mid | inventory_equal_share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.083668 | 0.828764 | 1.4398 | 106.525 | 0.013516 | 0.905661 | -65 | 0.005623 |

Full table: `reports/tables/strategy_similarity_diagnostics.csv`

## Historical Test Contribution By Date

| strategy | date | pnl_contribution | turnover_contribution | fill_count | end_inventory | max_abs_inventory | avg_quoted_spread_ticks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| avellaneda_stoikov_mid | 2024-08-05 | -323.131 | 796,557.13 | 12178 | 3,222 | 35,193 | 120.271 |
| avellaneda_stoikov_mid | 2024-08-06 | -70.746 | 392,319.46 | 5288 | 3,620 | 38,599 | 92.781 |
| avellaneda_stoikov_microprice | 2024-08-05 | -320.33 | 794,366.18 | 12155 | 752 | 38,453 | 120.284 |
| avellaneda_stoikov_microprice | 2024-08-06 | -72.642 | 389,275.31 | 5246 | -1,279 | 35,851 | 92.775 |

Full table: `reports/tables/historical_experiment_by_date.csv`

## Monte Carlo Simulation

| strategy | mean_pnl | std_pnl | p05_pnl | p95_pnl | mean_final_inventory | mean_turnover |
| --- | --- | --- | --- | --- | --- | --- |
| as_inventory_aware | 4.5922 | 4.5723 | -2.1206 | 12.065 | 0.045 | 726.004 |
| symmetric_mid_quotes | 4.7898 | 6.1204 | -4.2875 | 15.08 | -0.114 | 701.825 |

Full table: `reports/tables/monte_carlo_summary.csv`

## Figures

![avellaneda_stoikov_microprice_test_quotes](reports/figures/avellaneda_stoikov_microprice_test_quotes.png)

![avellaneda_stoikov_mid_test_quotes](reports/figures/avellaneda_stoikov_mid_test_quotes.png)

![historical_inventory_comparison](reports/figures/historical_inventory_comparison.png)

![historical_pnl_comparison](reports/figures/historical_pnl_comparison.png)

![microprice_adjustment_by_imbalance](reports/figures/microprice_adjustment_by_imbalance.png)

![microprice_signal_by_imbalance](reports/figures/microprice_signal_by_imbalance.png)

![monte_carlo_pnl_distribution](reports/figures/monte_carlo_pnl_distribution.png)

![quote_refresh_sensitivity](reports/figures/quote_refresh_sensitivity.png)

![validation_score_ranking](reports/figures/validation_score_ranking.png)

![volatility_sensitivity](reports/figures/volatility_sensitivity.png)

## Limitations

- `visible_size` partial fills cap quantity by top-of-book displayed size, but this is not a queue-position model.
- No latency model.
- No transaction fees or rebates in the final baseline.
- Trades are not used for baseline fills because the `side` convention is not independently documented.
- No forced terminal liquidation; final inventory may be non-zero.

## Improvement Roadmap

- Queue-position partial fills.
- Trade-based execution after stronger feed synchronization checks.
- More robust walk-forward calibration of `k`, volatility, and risk parameters.
- Rolling microprice recalibration and L2 depth features.
- Fees, rebates, latency, and forced liquidation.