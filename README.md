# CMF HFT Market-Making Backtester

Author: **Sergei Smirnov**

Educational Python research project for replaying historical limit order book data and
evaluating high-frequency market-making strategies.

The project was built for the CMF MSU entrance exam assignment:

- implement an integrated LOB replay backtester;
- support limit order placement, cancellation, and execution modeling;
- report PnL, inventory, turnover, and fill metrics;
- implement Avellaneda-Stoikov (2008);
- enhance it with Stoikov (2018) microprice;
- run historical and simulation experiments;
- provide configs, sample data, performance report, and technical documentation.

## What Is Implemented

- Event-driven LOB snapshot replay backtester.
- Limit order placement and cancel/replace workflow.
- Crossing-based execution model:
  - buy limit fills when future best ask crosses the bid;
  - sell limit fills when future best bid crosses the ask.
- Partial-fill approximation through visible top-of-book size.
- Portfolio accounting: cash, inventory, turnover, fills, mark-to-market PnL.
- Avellaneda-Stoikov market making with mid-price reference.
- Finite-state Stoikov microprice estimator.
- Avellaneda-Stoikov strategy with microprice reference.
- Chronological train/validation/test split.
- Parameter calibration and sensitivity analysis.
- Synthetic Monte Carlo experiment.
- Performance report with tables and figures.
- Unit and smoke tests.

## Strategy Models

The baseline strategy uses the Avellaneda-Stoikov reservation price and spread:

```text
r_t = reference_t - q_t * gamma * sigma_t^2 * tau
spread_t = gamma * sigma_t^2 * tau + 2/gamma * log(1 + gamma/k)
bid_t = r_t - spread_t / 2
ask_t = r_t + spread_t / 2
```

Two reference prices are compared:

```text
avellaneda_stoikov_mid:
  reference_t = mid_t

avellaneda_stoikov_microprice:
  reference_t = microprice_t = mid_t + G*(imbalance_t, spread_t)
```

The microprice adjustment `G*` is fitted on the train split using a finite-state
Markov model over imbalance buckets and spread states.

## Repository Layout

```text
configs/                 YAML experiment configs
data/sample/             small reproducible sample dataset
docs/                    model description, technical docs, research notes, roadmap
reports/                 generated performance report, tables, figures
src/cmf_backtester/
  backtest/              event loop and Numba backtest kernel
  calibration/           volatility, k diagnostic, validation grid
  data/                  loading, preprocessing, splitting
  execution/             orders and crossing execution model
  experiments/           Monte Carlo simulation
  market/                LOB features and microprice estimator
  portfolio/             accounting and performance metrics
  reporting/             plots and markdown report generation
  strategies/            Avellaneda-Stoikov strategies
tests/                   unit and smoke tests
```

## Working With Agents

The repository includes lightweight project memory for coding and research agents:

- [AGENTS.md](AGENTS.md): operating manual, safe commands, quant invariants.
- [CONTEXT.md](CONTEXT.md): stable project context and economic interpretation.
- [Agent memory](docs/agent_memory.md): durable facts and open research directions.
- [Research workspace](docs/research/README.md): source registry, local paper storage,
  and paper/experiment note templates.
- [Architecture decisions](docs/decisions/README.md): ADRs for stable design choices.
- [Agent playbooks](docs/agent_playbooks/quant_research_workflow.md): reusable workflows
  for quant research and performance optimization.

## Data

The repository includes a small sample dataset:

```text
data/sample/lob_sample.parquet
```

Large raw and processed market data files are intentionally excluded from git:

```text
data/raw/
data/processed/
reports/tables/*_timeseries.parquet
```

This keeps the repository lightweight. Full historical experiments can be reproduced
when the original raw LOB/trades data is placed under `data/raw/`.

Expected raw LOB source:

```text
data/raw/lob.parquet
```

Expected trades source, currently used only for diagnostics/roadmap:

```text
data/raw/trades.parquet
```

## Liquidation EDA Notebook

The repository also contains a separate full-data exploratory analysis for the
liquidation research task:

- [Notebook report](notebooks/liquidation_eda.ipynb)
- [Markdown report](reports/liquidation_eda/liquidation_eda_report.md)
- [EDA config](configs/liquidation_eda.yaml)
- [Research hypotheses](docs/research/notes/liquidation_signal_hypotheses.md)

The notebook studies Binance trades, Binance BBO, Binance liquidations, and
Bybit liquidations. It checks timestamp and side conventions, duplicate
timestamps, BBO quality, trade-to-BBO alignment, OFI, queue imbalance,
full-data maker markouts, liquidation context, signed-flow response functions,
and liquidation event studies.

The latest full run processed:

```text
Binance trades: 1,107,782,898 rows
Binance BBO:      206,966,513 rows
```

Open the notebook in GitHub preview, VS Code, JupyterLab, or any standard
notebook viewer. The executed tables and figures are saved in the notebook.

Regenerate the full EDA artifacts when the raw liquidation data is available:

```bash
PYTHONPATH=src .venv/bin/python -m cmf_backtester.main run-liquidation-eda \
  --config configs/liquidation_eda.yaml --profile full
```

The full pipeline is intentionally separated from the notebook because the
full-data run is expensive. The notebook loads the generated CSV tables and PNG
figures from `reports/liquidation_eda/` and preserves executed outputs for
review.

## Installation

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Run tests:

```bash
PYTHONPATH=src .venv/bin/python -m pytest
```

Current test status:

```text
27 passed
```

## Reproducing The Pipeline

Run the full local pipeline when raw data is available:

```bash
PYTHONPATH=src .venv/bin/python -m cmf_backtester.main preprocess-lob --config configs/as_mid.yaml
PYTHONPATH=src .venv/bin/python -m cmf_backtester.main fit-microprice --config configs/as_microprice.yaml
PYTHONPATH=src .venv/bin/python -m cmf_backtester.main estimate-k --config configs/as_mid.yaml
PYTHONPATH=src .venv/bin/python -m cmf_backtester.main run-sensitivity --config configs/validation_grid.yaml
PYTHONPATH=src .venv/bin/python -m cmf_backtester.main run-microprice-move-sensitivity
PYTHONPATH=src .venv/bin/python -m cmf_backtester.main run-quote-refresh-sensitivity
PYTHONPATH=src .venv/bin/python -m cmf_backtester.main run-volatility-sensitivity
PYTHONPATH=src .venv/bin/python -m cmf_backtester.main diagnose-microprice-signal
PYTHONPATH=src .venv/bin/python -m cmf_backtester.main run-historical-experiments
PYTHONPATH=src .venv/bin/python -m cmf_backtester.main diagnose-strategy-similarity
PYTHONPATH=src .venv/bin/python -m cmf_backtester.main run-monte-carlo --config configs/monte_carlo.yaml
PYTHONPATH=src .venv/bin/python -m cmf_backtester.main make-report
```

Single-strategy runs:

```bash
PYTHONPATH=src .venv/bin/python -m cmf_backtester.main run-backtest --config configs/as_mid.yaml
PYTHONPATH=src .venv/bin/python -m cmf_backtester.main run-backtest --config configs/as_microprice.yaml
```

## Final Historical Results

Final test split results under `visible_size` partial-fill approximation:

| Strategy | Final PnL | Max Drawdown | Turnover | Fills | Final Inventory | Max Abs Inventory |
|---|---:|---:|---:|---:|---:|---:|
| Avellaneda-Stoikov mid | -393.877 | 395.110 | 1,188,876.59 | 17,466 | 3,620 | 38,599 |
| Avellaneda-Stoikov microprice | -392.971 | 394.084 | 1,183,641.49 | 17,401 | -1,279 | 38,453 |

Interpretation:

- Both strategies are unprofitable under the conservative crossing-based execution
  assumption.
- This execution model is adverse-selection-heavy because fills occur when future
  best bid/ask crosses our quote level.
- The microprice strategy gives a small improvement in final PnL and drawdown.
- Microprice corrections are directionally meaningful but small relative to the
  Avellaneda-Stoikov quoted spread.

Detailed outputs:

- [Performance report](reports/performance_report.md)
- [Technical documentation](docs/technical_documentation.md)
- [Model description](docs/model_description.md)
- [Research audit](docs/research_audit.md)
- [Improvement roadmap](docs/improvement_roadmap.md)
- [Agent instructions](AGENTS.md)
- [Research source registry](docs/research/source_registry.yaml)

## Key Assumptions

- L1 top-of-book replay is extracted from L2 snapshots.
- Orders placed at timestamp `t` cannot fill on the same snapshot.
- Execution uses future best bid/ask crossing.
- Final historical configs use `visible_size` partial fills:

```yaml
execution:
  fill_mode: "visible_size"
```

- `visible_size` caps fill quantity by displayed top-of-book size.
- No queue position model.
- No latency model.
- No fees or rebates in the final baseline.
- Trades are not used for baseline execution because the `side` convention is not
  independently documented.
- Terminal inventory is marked to mid-price; no forced liquidation is applied.

## Calibration

The project uses chronological splitting:

```text
Train:       2024-08-01 to 2024-08-03
Validation: 2024-08-04
Test:       2024-08-05 to 2024-08-06
```

Train is used for:

- microprice estimator fitting;
- empirical `k` diagnostic;
- data and volatility diagnostics.

Validation is used for:

- `gamma`;
- `k`;
- `tau_seconds`;
- `max_inventory`;
- `quote_refresh_seconds`;
- microprice move-filter sensitivity.

Test is used only for final out-of-sample reporting.

Selected baseline parameters:

```text
gamma = 0.0001
k = 0.025
tau_seconds = 180
order_size = 10000
max_inventory = 100000
quote_refresh_seconds = 0.25
microprice.max_mid_move_ticks = 10.0
```

## Performance And Engineering

The implementation uses:

- Polars for CSV/Parquet preprocessing and report tables;
- NumPy arrays for runtime market data;
- Numba for the main Avellaneda-Stoikov crossing backtest kernel;
- Matplotlib for static report figures;
- Pytest for unit and smoke tests.

## Roadmap

The current version is an educational research backtester. The next improvements would
make execution and market microstructure modeling more realistic:

| Area | Why It Matters | Implementation Direction |
|---|---|---|
| Queue position model | `visible_size` does not know whether our order is first or last in queue. | Track `queue_ahead`, reduce it with trades/cancellations, fill only after queue is depleted. |
| Trade-based execution | Pure crossing execution selects mostly adverse price-moving events. | Synchronize trades with LOB, verify `side` convention, fill passive orders when aggressive trades consume our level. |
| L2 depth features | L1 imbalance is noisy and ignores deeper liquidity. | Add top-N and distance-weighted depth imbalance; use L2 pressure in microprice states or strategy filters. |
| Fees, rebates, latency | High-frequency PnL is highly sensitive to costs and delays. | Report gross/net PnL, add maker/taker economics, delay order activation/cancellation. |
| Forced liquidation and risk controls | Residual inventory can hide risk. | Add terminal liquidation, drawdown limits, kill switch, and stronger inventory skew near limits. |
| Walk-forward calibration | Market microstructure is non-stationary. | Fit on previous N days, validate on the next day, test on the following day, report parameter stability. |

More detail is available in [Improvement roadmap](docs/improvement_roadmap.md).

## License

Educational exam project. No production trading use is intended.
