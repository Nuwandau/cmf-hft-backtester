# Liquidation Signal Hypotheses

Generated from the liquidation EDA pipeline.

## Empirical Pointers From Current EDA

- Source coverage spans 2025-12-01 through 2026-02-28 for both BTCUSDT and ETHUSDT.
- Binance trades have many duplicate timestamps, so same-timestamp ordering must not be inferred.
- Full-data BBO quality, markout, liquidation context, response, and event-study tables are computed with daily chunks.
- Deterministic samples are retained only for visual distribution plots.

- Strongest full-data liquidation-context bucket by weighted maker PnL: ethusdt buy 300s upward_pressure / same_direction_toxic_risk = 11.3889 bps.
- Weakest full-data liquidation-context bucket by weighted maker PnL: ethusdt sell 300s upward_pressure / opposite_direction = -10.2294 bps.

## Strong Candidates To Test Later

- Same-direction liquidation pressure may identify toxic maker trades, but the direction is not uniform across symbol/side/horizon and must be validated split-by-split.
- Bybit liquidation clusters may lead Binance adverse selection after the required 200ms delay.
- OFI and queue imbalance may help distinguish toxic flow from ordinary trade flow.
- Extreme signed liquidation pressure may be nonlinear: saturation or reversal should be modeled.
- The future filter should use only known-at-time liquidation/BBO/trade features and should evaluate kept turnover against the 500k USD/day constraint.

## Risks

- Full EDA stores compact aggregates, not every enriched trade row.
- Liquidation response is descriptive, not causal proof.
- Same-timestamp ordering remains ambiguous in public data.
- A full production signal still needs a separate feature-generation path that returns one filter value per trade.

## Evidence Pointer

See `reports/liquidation_eda/tables/markout_by_liquidation_context.csv`.
