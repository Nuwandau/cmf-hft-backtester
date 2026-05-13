# CMF HFT Market-Making Backtester

Author: Sergei Smirnov

Educational Python backtesting engine for replaying historical limit order book snapshots and
evaluating Avellaneda-Stoikov market-making strategies.

## Strategies

- `avellaneda_stoikov_mid`: Avellaneda-Stoikov 2008 with mid-price reference.
- `avellaneda_stoikov_microprice`: Avellaneda-Stoikov with Stoikov 2018 finite-state microprice.

## Execution Assumption

Baseline execution uses LOB snapshot crossing:

- buy limit fills if future best ask is less than or equal to our bid;
- sell limit fills if future best bid is greater than or equal to our ask;
- orders placed at timestamp `t` cannot fill at the same timestamp.

The final historical configs use zero fees, no latency, no queue position model, and
the `visible_size` partial-fill approximation:

```yaml
execution:
  fill_mode: "visible_size"
```

This caps fills by visible top-of-book size. It is not a queue model. A simpler
`fill_mode: "full"` is still supported for controlled comparisons.

## Install

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run

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
PYTHONPATH=src .venv/bin/python -m cmf_backtester.main run-backtest --config configs/as_mid.yaml
PYTHONPATH=src .venv/bin/python -m cmf_backtester.main run-backtest --config configs/as_microprice.yaml
PYTHONPATH=src .venv/bin/python -m cmf_backtester.main make-report
```

## Tests

```bash
PYTHONPATH=src .venv/bin/python -m pytest
```

## Performance

The project uses:

- Parquet raw/processed data for fast reproducible local IO;
- Polars for CSV/Parquet preprocessing;
- NumPy arrays for runtime data;
- Numba for the main Avellaneda-Stoikov crossing backtest kernel.

## Documentation

- [Technical documentation](docs/technical_documentation.md)
- [Model description](docs/model_description.md)
- [Improvement roadmap](docs/improvement_roadmap.md)
