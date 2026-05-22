# AGENTS.md

This file is the first stop for coding agents working on this repository. Keep
answers and edits focused on the project objective: an educational Python HFT
market-making backtester with Avellaneda-Stoikov and Stoikov microprice models.

## Project Map

- `src/cmf_backtester/data/`: data loading, raw LOB preprocessing, chronological splits.
- `src/cmf_backtester/market/`: L1 market features, tick handling, finite-state microprice.
- `src/cmf_backtester/execution/`: order objects and crossing-based fill logic.
- `src/cmf_backtester/portfolio/`: cash, inventory, turnover, PnL accounting.
- `src/cmf_backtester/strategies/`: Avellaneda-Stoikov mid-price and microprice strategies.
- `src/cmf_backtester/backtest/`: event loop, recorder, Numba accelerated backtest kernel.
- `src/cmf_backtester/calibration/`: volatility, empirical `k`, validation search.
- `src/cmf_backtester/reporting/`: figures and markdown performance report.
- `configs/`: YAML configs for strategies, validation, Monte Carlo.
- `docs/`: technical docs, model description, roadmap, research notes, decisions.
- `reports/`: generated report tables and figures.
- `tests/`: unit and smoke tests.

Read `CONTEXT.md` before making model, execution, or calibration changes.

## Non-Negotiable Quant Invariants

- Preserve chronological ordering. Do not randomize train/validation/test splits.
- Avoid look-ahead bias. Orders created at timestamp `t` must not fill on the same
  snapshot if the decision used that snapshot.
- Use future best-ask crossing for buy limits and future best-bid crossing for sell
  limits unless a task explicitly changes the execution model.
- Keep portfolio accounting signed:
  - buy: cash decreases, inventory increases;
  - sell: cash increases, inventory decreases.
- Treat raw market data as local-only. Do not commit `data/raw/`, `data/processed/`,
  hidden-test data, or copyrighted paper PDFs.
- Keep the two exam strategies explicit: Avellaneda-Stoikov with mid-price and
  Avellaneda-Stoikov with microprice. Put extra strategies in the roadmap unless
  the user asks to implement them.

## Standard Commands

Set the module path for all local commands:

```bash
PYTHONPATH=src .venv/bin/python -m pytest
```

Run one backtest:

```bash
PYTHONPATH=src .venv/bin/python -m cmf_backtester.main run-backtest --config configs/as_mid.yaml
PYTHONPATH=src .venv/bin/python -m cmf_backtester.main run-backtest --config configs/as_microprice.yaml
```

Regenerate the final report when raw data is available:

```bash
PYTHONPATH=src .venv/bin/python -m cmf_backtester.main run-historical-experiments
PYTHONPATH=src .venv/bin/python -m cmf_backtester.main make-report
```

## Editing Rules

- Prefer small, scoped edits that match existing module boundaries.
- Use Polars for large tabular data and Numba only where profiling or code structure
  justifies it.
- Do not rewrite generated reports unless the task requires report regeneration.
- If a file is already dirty, inspect the diff and preserve user changes.
- Add tests for execution/accounting/model changes; smoke-test report commands when
  touching reporting.

## Research Workflow

- Store local paper PDFs in `docs/research/papers/`; they are ignored by git.
- Track paper metadata in `docs/research/source_registry.yaml`.
- Put implementation notes in `docs/research/notes/`.
- For architecture choices, add a short ADR under `docs/decisions/`.
- When using web or GitHub sources, cite exact URLs in the final answer and, when
  durable project context is useful, add them to the source registry.

## Agent Handoff Checklist

Before ending substantial work, update the user with:

- files changed;
- commands run and whether they passed;
- generated outputs, if any;
- assumptions or limitations introduced;
- next recommended step.
