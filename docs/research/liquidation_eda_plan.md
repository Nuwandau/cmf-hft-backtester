# Liquidation Dataset EDA Plan

Status: draft for review

This document is the execution plan for the first week of work on the liquidation
signal task. The goal is pure exploration: build intuition about the four public
datasets before implementing a signal or ML model.

The plan is intentionally implementation-oriented so that a future coding agent
can execute it step by step without turning the project into notebook spaghetti.

## 1. Task Interpretation

The current task is not to build the final filter yet. It is to understand the
data well enough to later design a robust trade filter.

Input datasets:

```text
data/raw/liquidation_task/
  description.md
  data/
    binance_trades/perp_btcusdt.parquet
    binance_trades/perp_ethusdt.parquet
    binance_booktickers/perp_btcusdt.parquet
    binance_booktickers/perp_ethusdt.parquet
    binance_liquidations/perp_btcusdt.parquet
    binance_liquidations/perp_ethusdt.parquet
    bybit_liquidations/btcusdt.parquet
    bybit_liquidations/ethusdt.parquet
```

The public description states:

- timestamps are `int64` microseconds since UNIX epoch in UTC;
- Binance trades `side` is taker side;
- liquidation `side` is the liquidation order side;
- Bybit liquidation timestamps must be shifted forward by `+200 ms` before
  matching them to Binance decision times;
- final hidden-test output will be a binary filter for Binance trades for
  horizons `tau in {30s, 120s, 300s}`.

EDA must verify these conventions on concrete examples and report any ambiguity.

## 2. Non-Goals For This Phase

Do not build the final signal in this phase.

Do not train ML models in this phase.

Do not tune thresholds against validation score yet, except for exploratory
diagnostics that help understand the data.

Do not modify the existing Avellaneda-Stoikov backtester architecture unless a
shared utility is obviously reusable.

Do not commit raw liquidation data or large derived Parquet files.

## 3. Architecture Principles

Keep this as a clean add-on to the existing project.

Recommended module boundary:

```text
src/cmf_backtester/liquidation/
  __init__.py
  config.py        # typed config loading and normalized paths
  schema.py        # expected columns, dtypes, side constants
  io.py            # Polars scan/read helpers and schema validation
  features.py      # pure BBO, trade, liquidation, rolling-window feature transforms
  markout.py       # maker direction, future-mid join, PnL/markout formula
  eda.py           # aggregation functions for tables
  plots.py         # EDA-specific figures
  report.py        # markdown report assembly
  cli.py           # one-command orchestration
```

Add one CLI command to the existing `src/cmf_backtester/main.py`:

```bash
PYTHONPATH=src .venv/bin/python -m cmf_backtester.main run-liquidation-eda --config configs/liquidation_eda.yaml
```

`main.py` should only register the command and delegate to
`cmf_backtester.liquidation.cli`. No liquidation business logic should be written
directly in `main.py`.

Add one config:

```text
configs/liquidation_eda.yaml
```

Baseline config fields:

```yaml
raw_root: data/raw/liquidation_task
output_root: reports/liquidation_eda
processed_root: data/processed/liquidation_task
symbols: [btcusdt, ethusdt]
markout_horizons_seconds: [30, 120, 300]
eda_curve_horizons_seconds: [1, 5, 10, 30, 60, 120, 300]
short_flow_windows_ms: [100, 250, 500, 1000, 2000]
bybit_delay_us: 200000
maker_rebate_bps: 0.5
bbo_staleness_tolerance_us: 5000000
plot_sample_rows: 200000
event_study_top_n: 5000
profile: full
```

Keep generated outputs under:

```text
reports/liquidation_eda/
  liquidation_eda_report.md
  tables/
  figures/
```

Optional reusable processed files may go under:

```text
data/processed/liquidation_task/
```

These files must remain untracked because `data/processed/` is ignored.
The full report must be reproducible from raw Parquet files. Any processed files
are deterministic rebuildable caches, not hidden state.

## 4. Data Handling And Performance

The dataset is multi-GB, so full eager reads should be avoided unless a file is
known to fit comfortably.

Performance is a first-class requirement for this EDA. Every implementation step
should prefer the fastest correct approach that keeps the code readable and
reproducible. The goal is not premature micro-optimization, but avoiding obvious
large-data mistakes and making repeated research runs practical.

Use:

- Polars `scan_parquet` for large reads;
- projection pushdown: select only needed columns;
- predicate pushdown by symbol/date where possible;
- `join_asof` for trades-to-BBO and event-to-BBO alignment;
- streaming collection where supported;
- sampled frames for dense scatter/histogram plots;
- full-data aggregates for summary tables;
- stable `original_row_id` columns when duplicate timestamps may make event order
  ambiguous.
- categorical/string normalization only once near the IO boundary;
- compact numeric dtypes where safe;
- symbol-by-symbol execution for heavy joins;
- deterministic caches for expensive intermediate markout/event-study tables;
- vectorized expressions over row-wise Python functions;
- NumPy arrays or Numba only for tight numerical loops that are hard to express
  efficiently in Polars.

Avoid:

- pandas as the default large-data engine;
- Python loops over rows;
- loading both BTC and ETH full trades eagerly unless necessary;
- storing large intermediate Parquet outputs in git;
- mixing heavy joins, plotting, and markdown generation in the same function.
- repeated full scans of the same file inside one command;
- materializing dense event-study panels before filtering to top-N events;
- drawing plots from full multi-million-row frames when a stable sample or
  aggregate is enough.

Memory discipline:

- run symbol-by-symbol where possible;
- write small CSV summary tables;
- write compact plots;
- if intermediate feature tables are needed, write them to ignored
  `data/processed/liquidation_task/`.

### 4.1 Performance plan by computation type

| Computation | Preferred method | Notes |
| --- | --- | --- |
| Source scans | `pl.scan_parquet(...).select(...)` | Only project needed columns. |
| Date/symbol filters | Polars lazy predicates | Push filters into parquet scan. |
| Daily/hourly counts | Polars group-by on derived UTC date/hour | Collect small aggregate only. |
| Trades to BBO | sorted Polars `join_asof` by symbol | Use tolerance and keep BBO age. |
| Future markout mid | `target_timestamp = timestamp + tau` plus backward as-of join | Do symbol-by-symbol and horizon-by-horizon if memory requires. |
| Rolling liquidation pressure | Polars dynamic windows where feasible | Cache compact per-symbol feature tables if expensive. |
| Event studies | top-N events plus sampled/control windows | Avoid full dense panels for all events. |
| Heavy custom loops | NumPy/Numba | Only after a clear bottleneck is identified. |
| Plots | aggregated data or deterministic samples | Avoid plotting raw full trades/BBO. |

### 4.2 Profiling and runtime metadata

The EDA command should record enough metadata to understand runtime and reproduce
results:

```text
reports/liquidation_eda/run_metadata.json
```

Include:

- command and config path;
- profile: `quick` or `full`;
- git commit if available;
- Python, Polars, NumPy versions;
- source file sizes and row counts;
- wall-clock runtime by phase;
- optional peak memory estimate if easy to collect;
- output files generated.

If a phase is slow enough to block iteration, add a short performance note to the
report or `docs/research/notes/liquidation_signal_hypotheses.md` explaining the
bottleneck and the chosen mitigation.

### 4.3 Quick vs full profile

The implementation should support two execution profiles:

```text
quick:
  small deterministic samples, top-N event studies, smoke-level plots

full:
  full-data aggregates, final EDA tables, publication-quality figures
```

The `quick` profile is for development and agent iteration. The `full` profile is
for final reporting.

## 5. Expected Report Structure

The final EDA report should be:

```text
reports/liquidation_eda/liquidation_eda_report.md
```

Suggested sections:

1. Objective and data sources.
2. Timestamp and side convention checks.
3. Dataset shape and coverage.
4. Data quality findings.
5. Binance BBO analysis.
6. Binance trades analysis.
7. Binance and Bybit liquidation analysis.
8. Cross-source alignment diagnostics.
9. Markout exploration.
10. Liquidation event studies.
11. Weird findings and anomalies.
12. Hypotheses for future signal design.
13. Next implementation steps.

The report should contain short interpretation paragraphs after each major table
or figure. It should not be a dump of plots.

## 6. Phase 0: Setup And Source Verification

### 6.0 Safe extraction check

If the archive has not been extracted yet:

- confirm enough free disk space;
- inspect the archive with `tar -tf` before extraction;
- extract into `data/raw/liquidation_task/`;
- verify that no extra nested directory level was created;
- keep raw tar/parquet files out of git.

In the current local workspace the data is already extracted to:

```text
data/raw/liquidation_task/
```

### 6.1 Verify local files

Check:

- `description.md` exists;
- all 8 parquet files exist;
- file sizes are non-zero;
- no accidental nesting problems after extraction.

Output:

```text
reports/liquidation_eda/tables/source_files.csv
```

Columns:

```text
source, venue, data_type, symbol, path, size_mb, exists
```

### 6.2 Load public task description into docs

Do not duplicate the full description verbatim in project docs. Instead:

- reference `data/raw/liquidation_task/description.md`;
- summarize the required conventions in the EDA report;
- keep all formulas used for markout in the report.

### 6.3 Confirm git hygiene

Ensure:

- `data/raw/` stays ignored;
- `data/processed/` stays ignored;
- `reports/liquidation_eda/cache/` stays ignored if created;
- `reports/liquidation_eda/tables/*.parquet`, if generated, are ignored or not staged;
- CSV, markdown, and PNG outputs may be tracked if not too large.
- `reports/liquidation_eda/run_metadata.json` is tracked because it documents
  reproducibility.

## 7. Phase 1: Schema, Timestamp, And Coverage Audit

### 7.1 Schema validation

For each file validate:

- exact columns;
- dtypes;
- ticker values;
- null counts;
- price and amount min/max;
- side unique values for trades/liquidations.

Expected schemas:

```text
Binance trades:
  timestamp, ticker, side, price, amount

Binance booktickers:
  timestamp, ticker, bid_price, bid_amount, ask_price, ask_amount

Binance liquidations:
  timestamp, ticker, side, price, amount

Bybit liquidations:
  timestamp, ticker, side, price, amount
```

Outputs:

```text
tables/schema_audit.csv
tables/nulls_and_ranges.csv
```

### 7.2 Timestamp sanity

For each file:

- min/max timestamp;
- converted min/max UTC datetime;
- row count;
- number of unique timestamps;
- duplicate timestamp count;
- duplicate full-row count;
- timestamp monotonicity violations;
- event-time gap distribution.
- same-timestamp row counts and whether `original_row_id` is needed for stable sorting.

Important checks:

- confirm microseconds by converting to UTC dates;
- confirm public split dates exist:
  - train: `2025-12-01` to `2026-01-31`;
  - validation: `2026-02-01` to `2026-02-28`;
- identify any missing days or large gaps.

Outputs:

```text
tables/time_coverage.csv
tables/time_gap_summary.csv
figures/event_counts_by_day.png
figures/event_counts_by_hour.png
figures/event_gap_distribution.png
```

### 7.3 Per-day and per-hour activity

Compute for each source/symbol:

- rows by day;
- rows by hour UTC;
- median/mean events per minute;
- burstiness proxy: p99 events per minute / median events per minute.

Research questions:

- Is activity uniform or bursty?
- Are there day/night patterns?
- Do liquidation events cluster in a few days?
- Does BTC have a different event intensity profile from ETH?

Outputs:

```text
tables/daily_event_counts.csv
tables/hourly_event_counts.csv
figures/event_counts_by_source_symbol.png
figures/intraday_activity_heatmap.png
```

### 7.4 Split-aware stability audit

All important descriptive statistics should be computed separately for:

```text
train:      2025-12-01 to 2026-01-31
validation: 2026-02-01 to 2026-02-28
```

For each split compare:

- event counts;
- spread and BBO liquidity;
- trade notional and side balance;
- liquidation notional and side balance;
- baseline maker markout;
- liquidation-conditioned markout.

Outputs:

```text
tables/train_validation_drift.csv
tables/daily_stability_summary.csv
figures/train_validation_activity_comparison.png
figures/daily_markout_stability.png
```

Acceptance rule: do not make a strong research conclusion from an effect that
only appears on one outlier day or only in one split.

## 8. Phase 2: Binance BBO / Bookticker EDA

### 8.1 Top-of-book sanity

For each symbol:

- count `bid_price <= 0` or `ask_price <= 0`;
- count `bid_amount < 0` or `ask_amount < 0`;
- count crossed states: `bid_price > ask_price`;
- count locked states: `bid_price == ask_price`;
- compute mid and spread:

```text
mid = 0.5 * (bid_price + ask_price)
spread = ask_price - bid_price
spread_bps = spread / mid * 10000
```

Outputs:

```text
tables/bbo_quality.csv
tables/spread_summary.csv
figures/spread_distribution_bps.png
figures/spread_by_day.png
```

### 8.2 BBO size and liquidity

Compute:

- bid amount distribution;
- ask amount distribution;
- top-of-book notional:

```text
bid_notional = bid_price * bid_amount
ask_notional = ask_price * ask_amount
```

- imbalance:

```text
bbo_imbalance = bid_amount / (bid_amount + ask_amount)
```

Outputs:

```text
tables/bbo_size_summary.csv
tables/bbo_imbalance_summary.csv
figures/bbo_size_distribution.png
figures/bbo_imbalance_distribution.png
figures/bbo_imbalance_by_hour.png
```

### 8.3 Mid-price dynamics

Compute:

- mid returns in bps over event time;
- mid returns over clock windows: 1s, 10s, 60s if feasible;
- realized volatility by hour/day;
- large mid jumps.

Research questions:

- Which days are volatile?
- Are liquidations concentrated during volatile windows?
- Are there obvious bad ticks?

Outputs:

```text
tables/mid_return_summary.csv
tables/large_mid_jumps.csv
figures/mid_price_by_symbol.png
figures/realized_volatility_by_day.png
```

### 8.4 Literature-informed BBO diagnostics: OFI and queue imbalance

The local papers by Cont, Kukanov, and Stoikov and by Gould and Bonart suggest
two additional diagnostics that are feasible from Binance bookticker data.

First, compute a level-I order flow imbalance style feature from consecutive BBO
updates. For event `n`:

```text
ofi_n =
  1{bid_price_n >= bid_price_{n-1}} * bid_amount_n
  - 1{bid_price_n <= bid_price_{n-1}} * bid_amount_{n-1}
  - 1{ask_price_n <= ask_price_{n-1}} * ask_amount_n
  + 1{ask_price_n >= ask_price_{n-1}} * ask_amount_{n-1}
```

Then aggregate OFI over short windows and compare it with future mid returns and
maker markouts.

Second, bucket queue imbalance:

```text
queue_imbalance = bid_amount / (bid_amount + ask_amount)
```

and estimate empirical probabilities of the next mid move being upward/downward
by bucket.

Outputs:

```text
tables/bbo_ofi_summary.csv
tables/queue_imbalance_next_move.csv
figures/ofi_vs_future_return.png
figures/queue_imbalance_next_move_probability.png
```

## 9. Phase 3: Binance Trades EDA

### 9.1 Trade size and notional

For each symbol and side:

- row count;
- amount distribution;
- price distribution;
- notional distribution:

```text
notional = price * amount
clipped_notional = min(notional, 100000)
```

- tail events: top 20 trades by notional.

Outputs:

```text
tables/trade_summary.csv
tables/top_trade_outliers.csv
figures/trade_amount_distribution.png
figures/trade_notional_distribution.png
figures/trade_side_counts.png
```

### 9.2 Side convention verification

The description says trade `side` is taker side:

```text
buy  => taker bought, maker sold
sell => taker sold, maker bought
```

Verify by joining each trade to the latest Binance BBO before the trade:

```text
trade_at_or_above_ask = trade_price >= prev_ask_price
trade_at_or_below_bid = trade_price <= prev_bid_price
```

Expected pattern:

- `buy` trades should mostly print near/above ask;
- `sell` trades should mostly print near/below bid.

Classify every trade relative to previous BBO:

```text
above_ask
at_ask
inside_spread
at_bid
below_bid
outside_or_ambiguous
```

Also compare two same-timestamp conventions:

```text
BBO before or equal trade time: bbo_ts <= trade_ts
BBO strictly before trade time: bbo_ts < trade_ts
```

If many trades and BBO updates share the same timestamp, the report must state
that event order inside the timestamp is not independently observable.

Produce concrete hand examples:

- timestamp UTC;
- symbol;
- side;
- trade price;
- previous bid/ask;
- inferred maker direction;
- future mid at 30s;
- markout sign.

Outputs:

```text
tables/trade_side_bbo_diagnostic.csv
tables/convention_examples_trades.csv
tables/trade_price_location_summary.csv
figures/trade_price_vs_bbo_position.png
```

### 9.3 Trade flow imbalance

Compute in rolling windows:

```text
buy_notional
sell_notional
signed_taker_notional = buy_notional - sell_notional
trade_count
```

Window candidates:

```text
100ms, 250ms, 500ms, 1s, 2s, 5s, 30s, 120s, 300s
```

Also compute signed trade-flow autocorrelation to check order-flow persistence.

Outputs:

```text
tables/trade_flow_summary.csv
tables/trade_flow_autocorrelation.csv
figures/signed_trade_flow_by_day.png
figures/trade_flow_autocorrelation.png
```

## 10. Phase 4: Liquidation EDA

### 10.1 Binance and Bybit liquidation distributions

For each venue/symbol/side:

- count;
- amount distribution;
- notional distribution;
- clipped notional distribution;
- top 20 liquidation events by notional;
- daily and hourly counts/notional.

Outputs:

```text
tables/liquidation_summary.csv
tables/top_liquidation_events.csv
figures/liquidation_notional_distribution.png
figures/liquidation_counts_by_day.png
figures/liquidation_notional_by_day.png
figures/liquidation_side_balance.png
```

### 10.2 Liquidation side convention verification

The description says liquidation `side` is order side:

```text
buy  => forced buy, short liquidation, upward pressure
sell => forced sell, long liquidation, downward pressure
```

Verify empirically:

- join liquidation event to Binance BBO before and after event;
- compute forward mid changes at `1s`, `5s`, `30s`, `120s`, `300s`;
- compare average signed move:

```text
signed_liq_direction = +1 for buy liquidation, -1 for sell liquidation
signed_forward_move_bps = signed_liq_direction * (future_mid - event_mid) / event_mid * 10000
```

Expected but not guaranteed:

- positive signed move after large liquidation clusters if liquidation flow contains
  information or coincides with momentum;
- possible reversal if liquidations mark exhaustion.

This is a sanity diagnostic, not a proof of the side convention: liquidation
pressure can be followed by either continuation or reversal depending on regime.

Outputs:

```text
tables/liquidation_side_markout_diagnostic.csv
tables/convention_examples_liquidations.csv
figures/liquidation_signed_forward_move.png
```

### 10.3 Liquidation clustering

Build rolling liquidation features by venue/symbol:

```text
liq_count_100ms, liq_count_250ms, liq_count_500ms
liq_count_1s, liq_count_5s, liq_count_30s, liq_count_120s, liq_count_300s
buy_liq_notional_window
sell_liq_notional_window
signed_liq_notional_window = buy_liq_notional - sell_liq_notional
abs_liq_notional_window = buy_liq_notional + sell_liq_notional
```

Also compute signed liquidation-flow autocorrelation and cluster duration
statistics. This follows the order-flow persistence warning in the literature:
liquidation clusters may be a state variable, not isolated independent events.

For Bybit, create both:

```text
raw timestamp features
available timestamp features = timestamp + 200ms
```

Outputs:

```text
tables/liquidation_cluster_summary.csv
tables/liquidation_flow_autocorrelation.csv
figures/liquidation_cluster_examples.png
figures/liquidation_flow_autocorrelation.png
```

## 11. Phase 5: Cross-Source Alignment

### 11.1 Trades to BBO

Use as-of join:

```text
trade timestamp -> latest BBO at or before trade timestamp
```

Compute:

- trade position relative to BBO:

```text
price_minus_mid_bps
price_minus_bid_bps
price_minus_ask_bps
```

- spread at trade time;
- BBO age at trade time:

```text
bbo_age_us = trade_timestamp - bbo_timestamp
```

Outputs:

```text
tables/trade_bbo_alignment_summary.csv
figures/bbo_age_distribution_at_trades.png
```

### 11.2 Liquidations to BBO

For Binance liquidations:

```text
liquidation timestamp -> latest Binance BBO
```

For Bybit liquidations:

```text
available_timestamp = timestamp + 200_000 microseconds
available_timestamp -> latest Binance BBO
```

Compute:

- BBO age;
- spread at event time;
- mid before/after.

Outputs:

```text
tables/liquidation_bbo_alignment_summary.csv
figures/liquidation_bbo_age_distribution.png
```

### 11.3 Cross-exchange lead-lag

Explore whether Bybit liquidations line up with Binance activity:

- count Binance trades in windows after Bybit liquidation:

```text
[0s, 1s], [0s, 5s], [0s, 30s], [0s, 120s]
```

- compare to same-length pre-event windows;
- compare raw Bybit timestamp vs +200ms available timestamp;
- compute Binance mid move after Bybit liquidation by side and size bucket.
- run an exploratory lag grid only for diagnostics:

```text
-1000ms, -500ms, 0ms, +200ms, +500ms, +1000ms, +2000ms
```

Production features for the future filter must use `available_timestamp =
timestamp + 200_000`, not the best-looking lag.

Outputs:

```text
tables/bybit_binance_lead_lag_summary.csv
tables/bybit_delay_sensitivity.csv
figures/bybit_liq_binance_trade_intensity_event_study.png
figures/bybit_liq_binance_mid_event_study.png
```

### 11.3.1 Signed-flow response functions

Following the market-impact and cross-impact literature, compute descriptive
response functions rather than only unconditional event studies:

```text
R_flow,asset(tau) = E[signed_flow_t * return_{asset,t->t+tau}]
```

Use this for:

- Binance signed trade flow -> Binance BTC/ETH returns;
- Binance signed liquidation flow -> Binance BTC/ETH returns;
- Bybit signed liquidation flow, available after `+200ms`, -> Binance BTC/ETH returns;
- BTC signed liquidation pressure -> ETH returns;
- ETH signed liquidation pressure -> BTC returns.

This is descriptive EDA, not a causal estimate. Report it by train/validation and
by day where feasible.

Outputs:

```text
tables/signed_flow_response_functions.csv
figures/signed_flow_response_functions.png
```

### 11.4 Same-timestamp and as-of join sensitivity

Every as-of join must specify:

- join direction;
- join key;
- sorting key;
- tolerance;
- stale-row exclusion rule.

For known-at-time features:

```text
sort by symbol/ticker and timestamp
join backward to BBO at or before decision timestamp
exclude if bbo_age_us > bbo_staleness_tolerance_us
```

For markout:

```text
target_timestamp = trade_timestamp + tau
join backward to BBO at or before target_timestamp
exclude if future_bbo_age_us > bbo_staleness_tolerance_us
```

Add diagnostics for:

- equal-timestamp joins;
- strictly-before joins;
- duplicate BBO updates at the same timestamp;
- duplicate trades at the same timestamp;
- whether preserving `original_row_id` changes any summary materially.

Outputs:

```text
tables/asof_join_sensitivity.csv
tables/same_timestamp_diagnostics.csv
```

## 12. Phase 6: Markout Exploration

This phase computes the target economics but still does not build a production
signal. It establishes the baseline maker economics that any future filter must
improve.

### 12.1 Markout formula

For each Binance trade `i`:

```text
p_i = trade price
m_i(tau) = Binance BBO mid at t_i + tau, forward-filled
s_i = +1 if taker buy, so maker sells
s_i = -1 if taker sell, so maker buys
w_i = min(price_i * amount_i, 100000)
maker_rebate_bps = config.maker_rebate_bps

pnl_i(tau) = -s_i * (m_i(tau) - p_i) / p_i * 10000 + maker_rebate_bps
```

Compute for:

```text
task horizons: 30s, 120s, 300s
EDA curve horizons: 1s, 5s, 10s, 30s, 60s, 120s, 300s
```

Exclude trades whose `t_i + tau` is beyond the available BBO range.

Outputs:

```text
data/processed/liquidation_task/trade_markouts_<symbol>.parquet  # optional, ignored
tables/baseline_all_trades_markout.csv
tables/daily_weighted_markout.csv
tables/markout_summary.csv
tables/markout_confidence_intervals.csv
figures/markout_distribution_by_tau.png
figures/markout_by_side_symbol_tau.png
figures/markout_curve_by_side_symbol.png
```

Stability diagnostics:

- weighted mean and median markout;
- winsorized mean markout;
- daily distribution of weighted markout;
- bootstrap confidence intervals over days;
- share of total PnL explained by top 1% worst trades.

### 12.2 Markout by market regime

Group markouts by:

- symbol;
- side;
- hour UTC;
- spread bucket;
- BBO imbalance bucket;
- trade notional bucket;
- recent volatility bucket;
- recent trade-flow imbalance bucket.
- recent return/trend bucket.
- recent OFI bucket;
- queue-imbalance bucket.

Add nonlinear response diagnostics motivated by the crypto fragmentation paper:

- bin signed trade-flow and liquidation-pressure variables into quantile buckets;
- plot average future return/markout by bucket;
- compare raw signed notional, signed square-root notional, and signed log-notional
  transformations;
- check for saturation or reversal in extreme buckets.

Outputs:

```text
tables/markout_by_regime.csv
tables/markout_by_recent_return.csv
tables/nonlinear_flow_response.csv
figures/markout_by_spread_bucket.png
figures/markout_by_hour.png
figures/markout_by_trade_size_bucket.png
figures/nonlinear_flow_response.png
```

### 12.3 Markout around liquidations

For each trade, join recent liquidation features before the trade:

```text
binance_liq_notional_last_1s/5s/30s/120s/300s
bybit_liq_notional_last_1s/5s/30s/120s/300s using +200ms availability
signed_liq_notional_window
same_direction_liq_pressure
opposite_direction_liq_pressure
recent_return
recent_volatility
```

Important direction logic:

- taker buy trade means maker sell;
- buy liquidation is upward pressure;
- a maker sell during recent upward pressure may be toxic;
- taker sell trade means maker buy;
- sell liquidation is downward pressure;
- a maker buy during recent downward pressure may be toxic.

Outputs:

```text
tables/markout_by_liquidation_context.csv
figures/markout_after_liquidation_clusters.png
figures/markout_by_same_direction_liq_pressure.png
```

### 12.4 Turnover-aware exploratory frontiers

Do not tune the final signal in this EDA phase. However, compute descriptive
frontiers that show whether simple toxicity filters could plausibly satisfy the
future turnover constraint.

Examples:

- remove trades in top liquidation-pressure quantiles;
- remove trades during widest-spread quantiles;
- remove trades during high-volatility quantiles.

For each exploratory rule report:

```text
removed_trade_share
kept_clipped_turnover_per_day
weighted_pnl_all
weighted_pnl_kept
weighted_pnl_filtered
```

Outputs:

```text
tables/exploratory_filter_frontiers.csv
figures/turnover_pnl_frontier.png
```

## 13. Phase 7: Event Studies

### 13.1 Liquidation event study

For large liquidation events and clusters:

- align Binance mid around event time;
- normalize event time to zero;
- plot average signed mid move from `-300s` to `+300s`;
- separate:
  - Binance vs Bybit;
  - BTC vs ETH;
  - buy vs sell liquidation;
  - notional size buckets;
  - own-asset vs cross-asset response.

Cross-asset checks:

```text
BTC liquidation pressure -> ETH Binance mid/markout
ETH liquidation pressure -> BTC Binance mid/markout
market-wide liquidation pressure = BTC + ETH
```

Outputs:

```text
tables/liquidation_event_study_summary.csv
tables/cross_asset_event_study_summary.csv
figures/liquidation_event_study_mid.png
figures/liquidation_event_study_spread.png
figures/liquidation_event_study_trade_intensity.png
figures/cross_asset_liquidation_event_study.png
```

### 13.2 Trade event study

For large Binance trades:

- align mid, spread, and BBO imbalance around trade;
- separate taker buy vs taker sell;
- compare trades with and without recent liquidation pressure.

Outputs:

```text
tables/trade_event_study_summary.csv
figures/trade_event_study_mid.png
figures/trade_event_study_spread.png
```

## 14. Phase 8: Weirdness And Anomaly Investigation

Create an explicit anomaly log.

Look for:

- missing days;
- periods with zero BBO updates but active trades;
- duplicate timestamps with conflicting rows;
- crossed/locked BBO;
- trades far outside contemporaneous bid/ask;
- liquidation prices far from Binance BBO;
- extreme event bursts;
- Bybit liquidation timestamps that appear to lead/lag Binance implausibly;
- days where BTC and ETH behave very differently.

Outputs:

```text
tables/anomaly_log.csv
figures/anomaly_case_studies.png
```

Each anomaly row should include:

```text
symbol, source, timestamp_utc, issue_type, severity, notes
```

## 15. Phase 9: Hypothesis Notebook Without Notebook Spaghetti

Instead of ad hoc notebooks, write hypotheses into the EDA report and optionally
into:

```text
docs/research/notes/liquidation_signal_hypotheses.md
```

Hypothesis format:

```text
H1: Large same-direction liquidation pressure predicts negative maker markout.

Evidence:
  table/figure references

Possible future feature:
  signed_liq_notional_30s, same_direction_liq_pressure_30s

Risks:
  overfitting to liquidation clusters, hidden-test regime shift, low turnover
```

Minimum expected hypotheses:

- toxicity after same-direction liquidation pressure;
- cross-exchange Bybit liquidation lead-lag;
- wide-spread regime toxicity;
- high-volatility regime toxicity;
- trade-size tail toxicity;
- liquidation exhaustion/reversal possibility.
- liquidation pressure is more useful when BBO is weak: wide spread, thin size,
  or extreme imbalance;
- Bybit liquidations may matter mainly in clusters, not as isolated prints;
- liquidation effects may be horizon-dependent: momentum at short horizons and
  reversal at longer horizons;
- adverse selection may be concentrated in tail trades;
- BTC and ETH may transmit stress cross-asset;
- signal robustness may depend on high-volatility/high-liquidation regimes.
- BBO-derived OFI may explain short-horizon markout better than raw spread or
  queue imbalance alone;
- queue imbalance may be most useful for very short-horizon markout and next
  mid-move diagnostics;
- signed trade/liquidation flow may have nonlinear or saturated impact, so bucket
  diagnostics may be more informative than raw correlations.

## 16. Phase 10: Final EDA Deliverables

Minimum deliverables:

```text
reports/liquidation_eda/liquidation_eda_report.md
reports/liquidation_eda/tables/source_files.csv
reports/liquidation_eda/tables/schema_audit.csv
reports/liquidation_eda/tables/time_coverage.csv
reports/liquidation_eda/tables/daily_event_counts.csv
reports/liquidation_eda/tables/bbo_quality.csv
reports/liquidation_eda/tables/bbo_ofi_summary.csv
reports/liquidation_eda/tables/queue_imbalance_next_move.csv
reports/liquidation_eda/tables/trade_summary.csv
reports/liquidation_eda/tables/liquidation_summary.csv
reports/liquidation_eda/tables/trade_side_bbo_diagnostic.csv
reports/liquidation_eda/tables/markout_summary.csv
reports/liquidation_eda/tables/baseline_all_trades_markout.csv
reports/liquidation_eda/tables/daily_weighted_markout.csv
reports/liquidation_eda/tables/markout_by_liquidation_context.csv
reports/liquidation_eda/tables/train_validation_drift.csv
reports/liquidation_eda/tables/signed_flow_response_functions.csv
reports/liquidation_eda/tables/nonlinear_flow_response.csv
reports/liquidation_eda/tables/anomaly_log.csv
reports/liquidation_eda/run_metadata.json
reports/liquidation_eda/figures/event_counts_by_day.png
reports/liquidation_eda/figures/spread_distribution_bps.png
reports/liquidation_eda/figures/trade_notional_distribution.png
reports/liquidation_eda/figures/liquidation_notional_distribution.png
reports/liquidation_eda/figures/ofi_vs_future_return.png
reports/liquidation_eda/figures/queue_imbalance_next_move_probability.png
reports/liquidation_eda/figures/markout_distribution_by_tau.png
reports/liquidation_eda/figures/markout_curve_by_side_symbol.png
reports/liquidation_eda/figures/signed_flow_response_functions.png
reports/liquidation_eda/figures/nonlinear_flow_response.png
reports/liquidation_eda/figures/liquidation_event_study_mid.png
configs/liquidation_eda.yaml
docs/research/notes/liquidation_signal_hypotheses.md
```

Optional deliverables:

```text
data/processed/liquidation_task/trade_markouts_btcusdt.parquet
data/processed/liquidation_task/trade_markouts_ethusdt.parquet
reports/liquidation_eda/tables/bybit_binance_lead_lag_summary.csv
reports/liquidation_eda/figures/bybit_liq_binance_mid_event_study.png
```

## 17. Suggested Implementation Order

### Step 1: Add module skeleton

Create:

```text
src/cmf_backtester/liquidation/
  __init__.py
  config.py
  schema.py
  io.py
  features.py
  markout.py
  eda.py
  plots.py
  report.py
  cli.py
```

Keep all paths configurable but use sensible defaults pointing to
`data/raw/liquidation_task`.

### Step 2: Add basic source audit

Implement:

- source file discovery;
- schema audit;
- timestamp coverage audit;
- daily/hourly counts.
- split-aware stability tables.

Generate first version of report with only sections 1-4.

### Step 3: Add BBO EDA

Implement:

- BBO mid/spread/imbalance features;
- BBO-derived OFI and queue-imbalance next-move diagnostics;
- BBO quality tables;
- spread and liquidity plots.

### Step 4: Add trades EDA

Implement:

- trade notional and side summaries;
- trades-to-BBO as-of diagnostic;
- price-location classification relative to previous BBO;
- subsecond signed trade-flow windows and autocorrelation;
- convention examples.

### Step 5: Add liquidation EDA

Implement:

- liquidation notional summaries;
- side balance;
- cluster diagnostics;
- subsecond liquidation-pressure windows and autocorrelation;
- Bybit `+200ms` availability timestamp.

### Step 6: Add markout computation

Implement:

- BBO mid forward-fill at `trade_timestamp + tau`;
- maker PnL in bps;
- weighted summaries by tau/symbol/side.
- baseline maker economics by day and split;
- markout curves over exploratory horizons.

### Step 7: Add cross-source diagnostics

Implement:

- markout by recent liquidation pressure;
- Bybit/Binance lead-lag tables;
- same-timestamp/as-of sensitivity diagnostics;
- cross-asset liquidation diagnostics;
- signed-flow response functions;
- nonlinear flow/markout bucket diagnostics;
- event studies.

### Step 8: Add anomaly log and hypotheses

Implement:

- anomaly table;
- hypothesis notes;
- final interpretive report.

### Step 9: Add tests and smoke checks

Tests should use small synthetic frames, not the 5GB raw dataset.

Test:

- timestamp conversion;
- Bybit `+200ms` shift;
- Bybit lag diagnostics never replace the production availability timestamp;
- trade side to maker direction mapping;
- markout sign formula;
- as-of join behavior;
- stale BBO tolerance/exclusion;
- joins do not mix BTC and ETH;
- duplicate timestamp deterministic ordering;
- BBO spread/imbalance calculation;
- duplicate timestamp handling.

Suggested test file:

```text
tests/test_liquidation_features.py
tests/test_liquidation_markout.py
tests/test_liquidation_io.py
```

Smoke-test CLI on tiny synthetic Parquet data:

```bash
PYTHONPATH=src .venv/bin/python -m cmf_backtester.main run-liquidation-eda \
  --config tests/fixtures/liquidation_eda_test.yaml
```

The smoke fixture should use tiny synthetic Parquet files in `tmp_path` and must
not require the real 5GB dataset.

### Step 10: Documentation update

Update:

- `README.md` with one short link to the EDA report;
- `docs/research/source_registry.yaml` if new papers/repos are used;
- `AGENTS.md` only if a new stable command or workflow is introduced.

## 18. Quality Bar

EDA is acceptable only if it answers these questions clearly:

- What is the size and time coverage of every source?
- Are timestamps and side conventions verified?
- Are there duplicates, gaps, bad prices, crossed BBO states, or other anomalies?
- What are the main distributional differences between BTC and ETH?
- What are the main distributional differences between Binance and Bybit liquidations?
- What is the baseline maker markout by tau, symbol, side, and regime?
- Do liquidation events appear informative for future Binance markouts?
- Are effects stable across train and validation?
- Are conclusions robust to daily outliers and heavy tails?
- Is Bybit used only through available time after the `+200ms` delay?
- Are there plausible hypotheses for a future binary trade filter?
- Are all results reproducible from one CLI command?

## 19. Risks And Mitigations

| Risk | Why It Matters | Mitigation |
| --- | --- | --- |
| Memory blowups | Trades and BBO are large. | Symbol-by-symbol Polars lazy scans, projections, aggregated outputs. |
| Misreading `side` | Wrong maker PnL sign invalidates all results. | Concrete BBO examples and unit tests for side mapping. |
| Incorrect timestamp units | Breaks all joins and horizons. | UTC conversion audit and visible date-range checks. |
| Bybit delay ignored | Hidden-test feature timing would be look-ahead biased. | Store both raw and `available_timestamp`; use available time for features. |
| Asof join misuse | Could use future BBO by accident. | Tests verifying backward/forward join semantics. |
| Plot overload | Too many plots can obscure conclusions. | Require interpretation paragraphs and prioritize hypothesis-relevant figures. |
| Hidden-test overfit | EDA may lead to hand-tuned thresholds. | Keep EDA descriptive; reserve threshold selection for later validation plan. |
| Project clutter | New task could pollute the AS backtester. | Use isolated `cmf_backtester/liquidation/` module and separate reports directory. |

## 20. Open Questions To Resolve During EDA

- Are there days where Binance BBO coverage is insufficient for markouts?
- Do trades often print inside the spread, at bid/ask, or outside the book?
- Does Bybit liquidation flow lead Binance mid moves after the required `+200ms` delay?
- Are liquidation effects momentum-like or reversal-like?
- Does toxicity depend more on liquidation notional, liquidation count, or signed imbalance?
- Are BTC and ETH similar enough for shared features, or should signal logic be symbol-specific?
- Does the 500,000 USD/day turnover constraint look easy or restrictive under simple exploratory filters?
