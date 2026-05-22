# Performance Optimization Agent Playbook

Use this playbook before optimizing runtime-critical code.

## Preferred Stack

- Polars lazy scans for large Parquet reads, filtering, grouping, and joins.
- NumPy arrays for compact numerical kernels.
- Numba for tight loops with simple types and stable array shapes.
- Matplotlib for report figures unless a task explicitly needs interactive plots.

## Rules

- Profile or identify the bottleneck before adding complexity.
- Keep correctness tests around the optimized path.
- Preserve a readable Python path when practical, especially for educational logic.
- Avoid pushing raw data or generated heavy Parquet files.
- Check memory pressure before materializing multi-GB DataFrames.

## Typical HFT Hotspots

- as-of joins between trades and BBO;
- rolling realized volatility;
- crossing/fill simulation loops;
- finite-state transition counting;
- parameter grid search over many strategy runs.
