# Source Note: albers_cucuringu_howison_shestopaloff_2021_fragmentation

## Bibliographic Info

- Title: Fragmentation, Price Formation, and Cross-Impact in Bitcoin Markets
- Authors: Jakob Albers, Mihai Cucuringu, Sam Howison, Alexander Y. Shestopaloff
- Year: 2021
- Local file: `docs/research/papers/Fragmentation,_Price_Formation,_and_Cross_Impact_in_Bitcoin_Markets.pdf`
- Registry id: `albers_cucuringu_howison_shestopaloff_2021_fragmentation`

## Why It Matters

This is the closest paper to the new liquidation task because it studies crypto
market fragmentation, short-horizon price formation, cross-market information
flow, and maker adverse selection.

The paper argues that useful crypto signals can be built from short-lookback
microstructure features and from information that arrives asynchronously across
venues. That is directly aligned with Binance trades/BBO and Bybit/Binance
liquidation feeds.

## Implementation-Relevant Ideas

- Use short lookback windows, including subsecond horizons, not only 30s+ windows.
- Study cross-market lead-lag rather than only same-venue contemporaneous effects.
- Use signed trade flow in USD, not only relative buy/sell ratios.
- Check nonlinear/saturating relationships by binning feature values against future returns.
- Treat extreme trade-flow/liquidation-flow values carefully; they may represent forced liquidations or sloppy execution, not linearly increasing information.
- Use adverse-selection curves across many horizons to evaluate maker fills.
- Be explicit that realistic maker backtesting is hard without queue position data.

## How It Maps To This Project

- Add subsecond trade-flow and liquidation-pressure windows to EDA:
  `100ms`, `250ms`, `500ms`, `1s`, `2s`.
- Add binned response plots:
  future maker markout vs signed trade flow, signed liquidation pressure, and BBO imbalance.
- Add cross-exchange diagnostics:
  Bybit liquidation pressure to future Binance mid/markout.
- Add cross-asset diagnostics:
  BTC liquidation pressure to ETH markout and ETH to BTC markout.
- Add adverse-selection curves:
  maker PnL/markout at `1s`, `5s`, `10s`, `30s`, `60s`, `120s`, `300s`.

## EDA Consequences

- Do not rely only on linear correlations.
- Bucket and winsorize heavy-tailed signed flow variables.
- Report whether effects survive train/validation split, daily breakdowns, and outlier days.
- Treat liquidation clusters as a first-class object of study.

## Open Questions

- Are Bybit liquidation events true leaders for Binance, or merely contemporaneous stress markers?
- Is liquidation pressure more informative in wide-spread/thin-BBO regimes?
- Is the relationship between liquidation notional and maker markout monotone, saturated, or reversal-like?
