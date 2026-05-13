# Technical Documentation

## Objective

This project implements an educational HFT backtesting engine for market-making strategies.
The engine replays historical limit order book snapshots and evaluates strategy performance
under an explicit crossing-based execution assumption.

## Data Format

The baseline market replay source is `data/raw/lob.parquet`. It is converted from the
provided raw CSV and keeps the original row index as `event_id`.

Required LOB columns:

- `event_id`;
- `local_timestamp`;
- `asks[0].price`, `asks[0].amount`;
- `bids[0].price`, `bids[0].amount`;
- additional L2 levels are kept in raw data but not required by baseline execution.

The trades file `data/raw/trades.parquet` is not required for baseline execution. Its
`side` column is assumed to denote taker/aggressor side only for diagnostics.

## Explicit Assumptions

The project intentionally uses a transparent educational market simulator rather than a
production exchange simulator. The main assumptions are:

| Area | Assumption | Reason | Consequence |
| --- | --- | --- | --- |
| Market data | LOB snapshots are enough for L1 replay. | The assignment requires LOB simulation, not full incremental reconstruction. | Queue position and hidden liquidity are unknown. |
| Event identity | The raw unnamed CSV index is preserved as `event_id`. | Multiple market events can share a timestamp in other datasets. | Processed L1 data is sorted by `event_id`, preserving source event order. |
| Event ordering | Orders created at timestamp `t` can only fill on later snapshots. | Prevents look-ahead bias. | Same-snapshot fills are disallowed even if the quote crosses the current book. |
| Execution | A buy limit fills when future best ask crosses the order price; a sell limit fills when future best bid crosses. | Matches the exam statement. | Fills are adverse-selection-heavy relative to a Poisson fill model. |
| Fill size | Final configs use `visible_size`, which caps fills by top-of-book visible size; `full` is still available for comparisons. | Partial fills are optional in the assignment, but the final report uses the richer approximation. | `visible_size` is not a real queue model. |
| Fees | Baseline uses zero fees/rebates. | Keeps model focused on strategy mechanics. | Reported PnL is gross PnL. |
| Latency | No latency is modeled. | Snapshot timestamps are not enough to infer realistic venue latency. | Results are optimistic on reaction speed but pessimistic on crossing-only fills. |
| Trades | Trades are not used for baseline fills. | `side` convention is not documented by the data vendor in the provided files. | Trade-based execution is left for roadmap after venue-side verification. |
| Inventory | Portfolio inventory is stored in raw amount units; the AS reservation-price formula uses `portfolio_inventory / inventory_risk_unit`. | `q` in Avellaneda-Stoikov is signed inventory, while implementation units must be consistent with `gamma`, volatility, and order size. | One default fill changes the model inventory term by approximately one unit. |
| Terminal inventory | No forced liquidation is applied at the end of validation/test windows. | The assignment focuses on replay and strategy performance, not liquidation scheduling. | AS inventory skew controls risk but does not guarantee final inventory equals zero. |
| Microprice | Finite-state estimator is trained only on train split and filters large snapshot-to-snapshot mid-price jumps. | Stoikov microprice is local-state based; adjacent snapshots can contain accumulated moves. | Sparse or unreliable states fall back to mid-price; the jump filter is checked by sensitivity analysis. |

## Project Outputs

The main deliverables are:

| Deliverable | Path |
| --- | --- |
| Strategy source code | `src/cmf_backtester/strategies/avellaneda_stoikov.py`, `src/cmf_backtester/strategies/avellaneda_stoikov_microprice.py` |
| Backtest engine | `src/cmf_backtester/backtest/engine.py`, `src/cmf_backtester/backtest/kernels.py` |
| Execution model | `src/cmf_backtester/execution/execution_model.py` |
| Portfolio accounting | `src/cmf_backtester/portfolio/portfolio.py`, `src/cmf_backtester/portfolio/metrics.py` |
| Configs | `configs/as_mid.yaml`, `configs/as_microprice.yaml`, `configs/as_mid_partial.yaml`, `configs/validation_grid.yaml` |
| Raw Parquet source data | `data/raw/lob.parquet`, `data/raw/trades.parquet` |
| Data audit tables | `reports/tables/data_audit.csv`, `reports/tables/data_audit_by_date.csv` |
| Performance report | `reports/performance_report.md` |
| Research audit | `docs/research_audit.md` |
| Improvement roadmap | `docs/improvement_roadmap.md` |

## Preprocessing

Raw LOB Parquet is converted to compact L1 Parquet with Polars:

```bash
PYTHONPATH=src .venv/bin/python -m cmf_backtester.main preprocess-lob --config configs/as_mid.yaml
```

Generated features:

- source `event_id`;
- best bid and ask;
- top bid and ask size;
- price ticks;
- mid-price in ticks;
- spread in ticks;
- top-of-book imbalance;
- chronological split label.

Microprice fitting filters transitions with `abs(next_mid - mid) > max_mid_move_ticks`
in tick units. The initial conservative candidate was `1.0`, matching a local one-tick
interpretation. Because this is a strong assumption for snapshot data, the project runs
a robustness experiment over `max_mid_move_ticks in [1.0, 2.0, 5.0, 10.0]` and selects
the final value on validation.
States with fewer than `min_state_count` training observations fall back to zero
microprice adjustment.

## Feature Definitions

```text
mid_t = 0.5 * (best_bid_t + best_ask_t)
spread_t = best_ask_t - best_bid_t
imbalance_t = bid_size_t / (bid_size_t + ask_size_t)
```

Weighted mid-price diagnostic:

```text
weighted_mid_t = imbalance_t * best_ask_t + (1 - imbalance_t) * best_bid_t
weighted_mid_t - mid_t = (imbalance_t - 0.5) * spread_t
```

## Backtest Event Ordering

For each market snapshot at timestamp `t`:

1. load current market state;
2. match active orders created before `t`;
3. update portfolio cash, inventory, turnover, and fills;
4. mark portfolio to current mid-price;
5. pass market state and portfolio state to strategy;
6. cancel and place orders requested by strategy;
7. record metrics.

Orders placed at timestamp `t` cannot execute at the same timestamp. This prevents look-ahead bias.

## Execution Model

Baseline crossing rules:

```text
buy limit at p fills if future best_ask <= p
sell limit at p fills if future best_bid >= p
```

Fill modes:

```text
full:
  fill_qty = order.remaining_quantity

visible_size:
  buy fill_qty = min(order.remaining_quantity, current ask_size_0)
  sell fill_qty = min(order.remaining_quantity, current bid_size_0)
```

The `visible_size` mode is an optional partial-fill approximation. It is not a queue model,
because queue position is unknown in snapshot data.

## Portfolio Accounting

For buy fill:

```text
cash -= price * qty + fee
inventory += qty
```

For sell fill:

```text
cash += price * qty - fee
inventory -= qty
```

Mark-to-market PnL:

```text
pnl_t = cash_t + inventory_t * mid_t
```

## Avellaneda-Stoikov Inventory Units

The portfolio stores inventory in raw dataset amount units. In Avellaneda-Stoikov,
`q` is the signed inventory of the strategy. In this implementation the reservation-price
skew uses inventory expressed in strategy lots, so that `gamma`, volatility, and order
size remain on a stable numerical scale:

```text
q_model_t = portfolio_inventory_t / inventory_risk_unit
```

The default configs set `inventory_risk_unit = order_size`, so one full default fill changes
`q_model` by roughly one unit. Without this convention, a raw order size such as `10000` would
make `q * gamma * sigma^2 * tau` thousands of ticks wide for otherwise reasonable parameters.

## Parameter Choices

Default historical experiment parameters are stored in `configs/as_mid.yaml` and
`configs/as_microprice.yaml`.

| Parameter | Default | Role | Selection logic |
| --- | --- | --- | --- |
| `gamma` | `0.0001` | Inventory risk aversion. | Selected on validation grid by risk-adjusted score. |
| `k` | `0.025` | Exponential decay of fill/crossing intensity with quote distance. | Grid includes empirical train diagnostic range; selected on validation. |
| `tau_seconds` | `180.0` | Effective inventory-risk horizon. | Treated as practical constant horizon and selected on validation. |
| `order_size` | `10000.0` | Quantity per quote. | Fixed to a small, interpretable amount relative to observed L1 sizes. |
| `max_inventory` | `100000.0` | Hard inventory risk limit. | Selected on validation grid; `50000` was more restrictive but slightly worse by the risk-adjusted score. |
| `inventory_risk_unit` | `10000.0` | Converts raw inventory to model `q`. | Equal to one default order size. |
| `quote_refresh_seconds` | `0.25` | Minimum time between quote refreshes. | Selected on validation grid and checked by refresh sensitivity; values below the median snapshot interval act close to every-event refresh. |
| `min_spread_ticks` | `1` | Lower bound for quoted spread. | Exchange tick-level sanity bound. |
| `post_only` | `true` | Prevents crossing the current snapshot at quote placement time. | Keeps the strategy passive at placement. |
| `volatility.window_seconds` | `300.0` | Rolling volatility window. | Smooth short-horizon microstructure noise. |
| `volatility.floor_ticks_per_sqrt_second` | `0.1` | Volatility floor. | Prevents zero-risk quotes in quiet periods. |
| `microprice.imbalance_buckets` | `10` | Number of imbalance states. | Matches common finite-state microprice examples while keeping states populated. |
| `microprice.max_spread_state_ticks` | `10` | Spread-state cap. | Wider spreads share the overflow state. |
| `microprice.max_mid_move_ticks` | `10.0` | Filter for local price moves. | Selected by validation sensitivity over `1.0`, `2.0`, `5.0`, and `10.0`; it uses more transitions while still filtering very large snapshot-to-snapshot jumps. |
| `microprice.min_state_count` | `50` | Sparse-state fallback threshold. | Avoids noisy microprice corrections in rarely observed states. |

## Calibration Protocol

Splitting is chronological, never random:

| Split | Dates | Use |
| --- | --- | --- |
| Train | 2024-08-01 to 2024-08-03 | Tick diagnostics, volatility behavior, microprice estimator, empirical `k` diagnostic. |
| Validation | 2024-08-04 | Hyperparameter selection for `gamma`, `k`, `tau`, `max_inventory`, `quote_refresh_seconds`. |
| Test | 2024-08-05 to 2024-08-06 | Final out-of-sample historical performance report. |

The validation objective is:

```text
score = final_pnl - drawdown_penalty * max_drawdown - inventory_penalty * avg_abs_inventory
```

with defaults in `configs/validation_grid.yaml`.

The test set is used only after choosing parameters. The report also shows a by-date
test decomposition because 2024-08-05 has a materially wider-spread regime than the
train and validation days.

## Quote Refresh Sensitivity

Quote refresh controls how long stale quotes remain active before cancel/replace. In this
snapshot-crossing simulator, slower refresh generally increases exposure to adverse
crossings. The processed data has a median snapshot interval of roughly `0.515` seconds,
so refresh values below this level are close to every-event refresh.

The robustness command evaluates both AS variants over several refresh intervals:

```bash
PYTHONPATH=src .venv/bin/python -m cmf_backtester.main run-quote-refresh-sensitivity
```

Output:

- `reports/tables/quote_refresh_sensitivity.csv`;
- `reports/figures/quote_refresh_sensitivity.png`.

The final configs use `quote_refresh_seconds = 0.25`, selected on validation. This is a
fast cancel/replace assumption and should be read together with the no-latency limitation.

## Volatility Sensitivity

The AS reservation price and spread both depend on `sigma_t`, so the project includes a
separate robustness sweep over rolling volatility windows, volatility floors, and direct
sigma multipliers:

```bash
PYTHONPATH=src .venv/bin/python -m cmf_backtester.main run-volatility-sensitivity
```

Output:

- `reports/tables/volatility_sensitivity.csv`;
- `reports/figures/volatility_sensitivity.png`.

This table is not used to re-optimize the final test result. It documents how sensitive
PnL, drawdown, fill count, and quoted spread are to the volatility input.

## Microprice Signal Diagnostics

The microprice diagnostic checks whether the fitted adjustment is directionally consistent
with imbalance buckets and local next mid-price moves:

```bash
PYTHONPATH=src .venv/bin/python -m cmf_backtester.main diagnose-microprice-signal
```

Output:

- `reports/tables/microprice_signal_diagnostics.csv`;
- `reports/figures/microprice_signal_by_imbalance.png`.

The expected sanity pattern is negative adjustment for low bid imbalance, positive
adjustment for high bid imbalance, and the same directional pattern in local next mid
moves on train data.

## Strategy Similarity Diagnostics

The mid-price and microprice AS strategies can remain close in PnL even when the
microprice signal is directionally correct. The reason is scale and execution mechanics:
the microprice correction is still small relative to the AS quoted spread, while
crossing-based fills depend on whether future best bid/ask crosses the rounded quote.

The similarity diagnostic quantifies this effect:

```bash
PYTHONPATH=src .venv/bin/python -m cmf_backtester.main diagnose-strategy-similarity
```

Output:

- `reports/tables/strategy_similarity_diagnostics.csv`.

Important fields:

- `same_both_quotes_share`: fraction of events where rounded bid and ask quotes match;
- `p99_adjustment_to_spread_ratio`: 99th percentile absolute microprice adjustment
  divided by average AS quoted spread;
- `inventory_equal_share`: fraction of events with identical inventory paths.

## Strategy Source Code Mapping

The strategy deliverable is implemented in:

| Component | Path | Responsibility |
| --- | --- | --- |
| Base interface | `src/cmf_backtester/strategies/base.py` | Defines `on_market_update`. |
| Mid-price AS strategy | `src/cmf_backtester/strategies/avellaneda_stoikov.py` | Computes reservation price, total spread, bid/ask quotes, inventory limits. |
| Microprice AS strategy | `src/cmf_backtester/strategies/avellaneda_stoikov_microprice.py` | Reuses AS logic with `reference_t = microprice_t`. |
| Microprice estimator | `src/cmf_backtester/market/microprice.py` | Fits finite-state `G*(imbalance, spread)` on train data. |

The code is original project code. External Avellaneda-Stoikov repositories were used
as references for formula checks, inventory scaling intuition, and expected simulation
structure; their source code was not copied into this project.

## Runtime Performance

The implementation uses:

- Parquet raw/processed data for fast reproducible local IO;
- Polars for CSV/Parquet preprocessing;
- NumPy arrays for runtime data;
- Numba `@njit` for the main Avellaneda-Stoikov crossing backtest loop.

The debug Python loop remains available for tests and auditability.

## Empirical k Diagnostic

`k` can be checked on the train split by placing hypothetical quotes at multiple
distances from the current best bid/ask and measuring whether future best prices cross
those quotes within a fixed horizon:

```bash
PYTHONPATH=src .venv/bin/python -m cmf_backtester.main estimate-k --config configs/as_mid.yaml
```

For each horizon, the diagnostic fits:

```text
ln(lambda(delta)) = a - k * delta
```

where `lambda(delta) = -ln(1 - crossing_probability(delta)) / horizon_seconds`.
The implementation measures whether either side crosses within the horizon. This is a
calibration diagnostic for choosing a plausible `k` grid, not an exact per-side
Poisson-intensity estimator and not the baseline execution model.

## Microprice Move-Filter Sensitivity

The finite-state microprice estimator uses adjacent snapshot transitions. On this dataset,
many adjacent snapshots have multi-tick mid-price moves, so the local jump filter is an
important modeling decision rather than a harmless implementation detail.

The robustness command refits the estimator on train for several filters and evaluates the
resulting microprice strategy on validation and test:

```bash
PYTHONPATH=src .venv/bin/python -m cmf_backtester.main run-microprice-move-sensitivity
```

Output:

- `reports/tables/microprice_move_sensitivity.csv`;
- `data/processed/microprice_sensitivity/microprice_estimator_maxmove_*.npz`.

Interpretation:

- lower thresholds keep the estimator closer to local Stoikov-style one-tick dynamics;
- higher thresholds use more data but may mix local order-book information with coarser
  drift between snapshots;
- the final project uses `10.0` because it has the best partial-fill validation score
  among the tested filters and produces a stronger but still bounded microprice correction.
