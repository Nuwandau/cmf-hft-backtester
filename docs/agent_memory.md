# Agent Memory

This file stores durable project facts that are useful across future agent
sessions. Update it only for stable decisions, not for every temporary thought.

## Stable Project Decisions

- The exam project is implemented as a Python package under `src/cmf_backtester/`.
- The two required strategies are Avellaneda-Stoikov with mid-price and with finite-state microprice.
- The final historical report uses `visible_size` partial-fill approximation.
- Raw and processed market data stay local and are not committed.
- Research papers can be stored locally under `docs/research/papers/`, but only notes
  and metadata should be tracked in git.
- Large generated time series Parquet files are excluded from git; lightweight CSV
  summaries, figures, and markdown reports may be tracked.

## Open Research Directions

- EDA for the newly added liquidation-signal dataset.
- Trade-based execution diagnostics using documented taker-side conventions where available.
- Queue-position model using L2 snapshots and/or trade prints.
- Walk-forward microprice recalibration.
- More realistic fees, rebates, latency, and terminal inventory liquidation.

## Handoff Discipline

When an agent finishes a meaningful change, it should leave:

- a short summary in the final message;
- tests or commands run;
- any changed docs or generated reports;
- unresolved assumptions.
