# Project Context

This repository is an educational quantitative development project for the CMF
MSU HFT entrance assignment. The goal is not to emulate a production exchange,
but to demonstrate a rigorous event-driven market-making backtester and clear
microstructure reasoning.

## Core Economic Story

The strategy passively posts a bid and an ask. It earns spread when fills are
favorable, but is exposed to adverse selection when price moves through its
quotes. Avellaneda-Stoikov controls this by shifting quotes away from the side
that would increase risky inventory.

The baseline reference price is the mid-price. The enhanced strategy replaces it
with Stoikov's microprice, a data-driven estimate of the short-horizon fair price
based on top-of-book imbalance and spread state.

## Key Variables

- `best_bid`, `best_ask`: top-of-book prices from the LOB snapshot.
- `mid`: `(best_bid + best_ask) / 2`.
- `spread`: `best_ask - best_bid`.
- `imbalance`: `bid_size / (bid_size + ask_size)`.
- `inventory`: signed strategy position in raw amount units.
- `q_model`: inventory normalized by `inventory_risk_unit` for the AS formula.
- `sigma_t`: rolling realized volatility of mid-price changes in ticks per square-root second.
- `gamma`: inventory risk aversion.
- `k`: quote-distance sensitivity in the AS spread formula.
- `tau`: practical inventory-risk horizon, not necessarily literal session end time.

## Backtest Event Order

For each snapshot:

1. load current market state;
2. check fills for active orders created on earlier events;
3. update portfolio accounting;
4. call the strategy with current market and portfolio state;
5. cancel/replace quotes;
6. record metrics.

This order is intentionally conservative against look-ahead bias.

## Known Modeling Limits

- L1 replay uses best bid/ask extracted from LOB snapshots.
- Partial fills are approximated by visible top-of-book size.
- There is no queue position model, latency model, fee/rebate model, or hidden-liquidity model.
- Trades are not part of baseline execution; they are reserved for future trade-based validation.
- The current microprice effect is small relative to the AS spread under the conservative
  crossing execution model.

## Current Final Baseline

Final historical experiments use:

- `fill_mode: visible_size`;
- `gamma: 0.0001`;
- `k: 0.025`;
- `tau_seconds: 180.0`;
- `order_size: 10000.0`;
- `inventory_risk_unit: 10000.0`;
- `quote_refresh_seconds: 0.25`;
- `microprice.max_mid_move_ticks: 10.0`.

See `reports/performance_report.md` for current results.
