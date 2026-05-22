# Source Note: lillo_2021_order_flow_price_formation

## Bibliographic Info

- Title: Order flow and price formation
- Author: Fabrizio Lillo
- Year: 2021
- Local file: `docs/research/papers/Order flow and price formation.pdf`
- Registry id: `lillo_2021_order_flow_price_formation`

## Why It Matters

This paper is a compact review of order flow, market impact, cross-impact, and
price formation in LOB markets. It is useful for interpreting liquidation EDA
without making naive causal claims.

## Implementation-Relevant Ideas

- Order flow has strong temporal dependence; signed trade/liquidation flow should
  be studied with autocorrelation and clustering diagnostics.
- Market impact can be transient and horizon-dependent, so markout should be
  measured on a curve, not only one horizon.
- Cross-impact is a response of one asset or venue to order flow in another; this
  supports Bybit-to-Binance and BTC-to-ETH response-function diagnostics.
- Impact is often nonlinear/concave in signed volume; use signed sqrt/log and
  quantile buckets as descriptive transformations.
- Apparent impact can be confounded by recent returns and endogenous order flow.

## How It Maps To This Project

- Add signed trade-flow autocorrelation by symbol.
- Add liquidation-flow autocorrelation and cluster persistence.
- Add response functions:
  `E[signed_flow_t * future_return_{t+tau}]` for trades and liquidations.
- Condition liquidation effects on recent return, volatility, and signed trade flow.
- Report median/winsorized mean and bootstrap-by-day confidence intervals because
  market impact estimates are noisy and heavy-tailed.

## Open Questions

- Does liquidation flow contain new information, or does it only proxy for recent price moves?
- Do large liquidation clusters behave like metaorder/co-impact stress?
- Are cross-asset effects symmetric between BTC and ETH?
