# Improvement Roadmap

The implemented project satisfies the educational baseline: LOB replay, order placement
and cancellation, crossing-based execution, optional visible-size partial fills, metrics,
Avellaneda-Stoikov mid-price strategy, finite-state microprice extension, calibration,
historical experiment, Monte Carlo experiment, and reporting.

The items below are intentionally left as future extensions because each one requires
additional assumptions or richer data.

## Execution Realism

- Queue position model.
- Partial fills using queue depletion rather than visible top-of-book size.
- Trade-based execution after stronger LOB/trade synchronization.
- Latency model.
- Exchange fees and maker/taker rebates.
- Forced liquidation or terminal inventory penalty.
- More conservative execution model requiring quote level to be present before crossing,
  not only crossed by a later best price.

## Market Data

- Full order book reconstruction from incremental updates.
- More robust trade side verification from venue documentation.
- Multi-day walk-forward splits.
- L2 depth imbalance features.
- Crossed/locked-book cleaning policy for full-depth data, if future experiments use L2.

## Modeling

- Empirical calibration of `A` and `k` from crossing probabilities.
- Rolling volatility regime adaptation.
- Rolling microprice recalibration.
- Alternative microprice state spaces including L2 depth.
- More conservative fill assumptions under wide spreads.
- Terminal inventory penalty or liquidation rule for final PnL comparability.
- Side-specific `k_bid` and `k_ask` estimates instead of a single symmetric `k`.

## Engineering

- More extensive Numba kernels for calibration routines.
- Optional Cython only if profiling shows Numba is insufficient.
- More integration tests on sampled real data.
- CLI command for a complete one-shot reproducibility pipeline.
- Structured experiment registry with config hash and output manifest.
