# Model Description

## Avellaneda-Stoikov With Mid-Price

The baseline strategy uses the mid-price as the fair reference price:

```text
reference_t = mid_t
```

Reservation price:

```text
r_t = reference_t - q_t * gamma * sigma_t^2 * tau
```

In the implementation, portfolio inventory is stored in raw `amount` units, while the
`q_t` used in the Avellaneda-Stoikov formula is expressed in strategy inventory lots:

```text
q_t = portfolio_inventory_t / inventory_risk_unit
```

For the default historical experiments, `inventory_risk_unit = order_size`. This avoids
mixing model inventory units with large exchange amount units.

Approximate optimal total spread:

```text
spread_t = gamma * sigma_t^2 * tau + 2/gamma * log(1 + gamma/k)
```

Quotes:

```text
bid_t = r_t - spread_t / 2
ask_t = r_t + spread_t / 2
```

Inventory interpretation:

- long inventory lowers reservation price and encourages selling;
- short inventory raises reservation price and encourages buying;
- higher `gamma` increases inventory aversion;
- higher volatility widens quotes;
- `k` controls how quickly fill intensity decays with distance.

Implementation constraints:

- quotes are rounded to integer tick prices;
- `post_only = true` clips the bid to the current best bid and the ask to the current
  best ask, so the strategy does not intentionally cross the current snapshot;
- if inventory reaches `max_inventory`, the strategy stops quoting the side that would
  increase the risky inventory direction;
- the strategy cancels stale quotes before placing refreshed quotes.

Source code:

- `src/cmf_backtester/strategies/avellaneda_stoikov.py`;
- fast Numba path in `src/cmf_backtester/backtest/kernels.py`.

## Finite-State Microprice

The enhanced strategy replaces mid-price with Stoikov's microprice:

```text
P_micro_t = mid_t + G*(imbalance_t, spread_t)
```

State:

```text
X_t = (imbalance_bucket_t, spread_state_t)
```

Transition matrices:

- `Q_xy`: zero-mid-move transition from state `x` to state `y`;
- `R_xk`: non-zero mid-price jump probability from state `x`;
- `T_xy`: post-jump state transition after a non-zero mid-price move.

Adjustment:

```text
G1 = (I - Q)^(-1) R K
B = (I - Q)^(-1) T
G* = G1 + B G1 + B^2 G1 + ...
```

Implementation detail:

- the paper's example uses a small fixed `K`;
- this project builds the jump component from observed train-set mid-price changes, because snapshot data can contain multi-tick jumps.
- transitions with absolute mid-price moves larger than `max_mid_move_ticks` are filtered
  during fitting. The project checks `1.0`, `2.0`, `5.0`, and `10.0` ticks because the
  optimal tradeoff depends on snapshot frequency. The final config uses `10.0`, selected
  on validation.
- sparse states with fewer than `min_state_count` train observations fall back to zero
  adjustment, so the strategy uses the mid-price in poorly estimated states.

The microprice-enhanced strategy does not change the Avellaneda-Stoikov inventory term
or spread formula. It only replaces the fair-price reference:

```text
reference_t = P_micro_t
r_t = P_micro_t - q_t * gamma * sigma_t^2 * tau
```

Source code:

- `src/cmf_backtester/strategies/avellaneda_stoikov_microprice.py`;
- `src/cmf_backtester/market/microprice.py`.

On the current dataset the one-tick filter produced a very small microprice correction.
The partial-fill validation sensitivity selected `max_mid_move_ticks = 10.0`, which uses
more train transitions and produces a clearer imbalance-dependent correction while still
filtering very large snapshot-to-snapshot jumps. This remains a local fair-price correction,
not a standalone directional forecasting model.

## Calibration Summary

The project uses chronological calibration:

1. Train split estimates data diagnostics, microprice states, and empirical `k`.
2. Validation split selects `gamma`, `k`, `tau`, `max_inventory`, and quote refresh.
3. Test split is used once for final reporting.

The selected historical configuration is:

```text
gamma = 0.0001
k = 0.025
tau_seconds = 180
order_size = 10000
max_inventory = 100000
quote_refresh_seconds = 0.25
inventory_risk_unit = 10000
microprice.max_mid_move_ticks = 10.0
```

The empirical `k` diagnostic is intentionally treated as a sanity check for the grid,
not as a perfect estimator of the original paper's Poisson fill intensity.

## Synthetic Monte Carlo Experiment

The Monte Carlo experiment follows the 2008 paper's stylized assumptions:

- Brownian mid-price;
- exponential fill intensity:

```text
lambda(delta) = A exp(-k delta)
```

It compares:

- inventory-aware Avellaneda-Stoikov quoting;
- symmetric quoting around mid-price.

This experiment is supplementary. The main deliverable remains historical LOB replay.
