# Research Audit Notes

This note records the implementation checks against external references and the main
project risks found during the model audit.

## External References Checked

- Avellaneda-Stoikov formula check:
  [Hummingbot technical deep dive](https://hummingbot.org/blog/technical-deep-dive-into-the-avellaneda--stoikov-strategy/)
  states the finite-horizon reservation price and total spread as
  `r = s - q gamma sigma^2 (T-t)` and
  `spread = gamma sigma^2 (T-t) + 2/gamma * log(1 + gamma/kappa)`.
- Practical Avellaneda-Stoikov implementation:
  [Hummingbot strategy docs](https://hummingbot.org/strategies/v1-strategies/avellaneda-market-making/)
  describe reservation price, optimal spread, volatility, inventory target, and order-book
  liquidity estimators.
- Hummingbot source code:
  [avellaneda_market_making.pyx](https://github.com/hummingbot/hummingbot/blob/master/hummingbot/strategy/avellaneda_market_making/avellaneda_market_making.pyx)
  normalizes inventory relative to target/portfolio size. This supports using
  `inventory_risk_unit` in this project instead of raw LOB amount units inside the
  Avellaneda-Stoikov `q`.
- Simulation reference:
  [fedecaccia/avellaneda-stoikov](https://github.com/fedecaccia/avellaneda-stoikov)
  and [ragoragino/avellaneda-stoikov](https://github.com/ragoragino/avellaneda-stoikov)
  reproduce the original Monte Carlo setting with unit inventory changes and stylized
  Poisson fills.
- Additional educational Avellaneda-Stoikov implementations:
  [DYSIM/Avellaneda-Stoikov-Implementation](https://github.com/DYSIM/Avellaneda-Stoikov-Implementation)
  and [mdibo/Avellaneda-Stoikov](https://github.com/mdibo/Avellaneda-Stoikov) were
  checked as reference implementations of the original model. They are useful for
  comparing formulas and simulation structure, but they do not provide a full historical
  LOB replay engine.
- Microprice reference:
  [sstoikov/microprice](https://github.com/sstoikov/microprice) is the author's
  public notebook. The notebook estimates a finite-state microprice from spread and
  imbalance states and filters to local one-tick mid-price moves before building the
  transition matrices.

## Issues Found And Fixes

1. Inventory scale was too large for the model.
   The portfolio inventory is measured in raw exchange amount units, while the
   Avellaneda-Stoikov `q` should be a model inventory count. The strategy now uses
   `q_model = inventory / inventory_risk_unit`; default `inventory_risk_unit = order_size`.

2. Microprice was learning multi-snapshot jumps.
   The LOB snapshots are not every order-book event, so adjacent snapshots may contain
   large accumulated price moves. The finite-state estimator now filters transitions
   with `abs(delta_mid) > max_mid_move_ticks`; default is one tick.

3. Sparse microprice states were noisy.
   States with fewer than `min_state_count` train observations now use zero adjustment,
   which makes the strategy fall back to mid-price in poorly estimated states.

4. The original validation grid missed empirically plausible `k`.
   Train-set crossing diagnostics put `k` closer to the low range around `0.025-0.07`
   ticks^-1 depending on horizon. The validation grid now includes this range.

5. The test split contains a market-regime shift.
   The 2024-08-05 test day has much wider spreads than train/validation days. This is
   not a code bug, but it must be highlighted when interpreting final PnL.

## Interpretation

Historical PnL can remain negative even after the fixes. Under the project execution
assumption, a fill happens only after the future market best price crosses our quote.
That is an adverse-selection-heavy rule and differs from the original paper's synthetic
Poisson fill process, where passive fills are sampled from an intensity model.

The microprice strategy may be close to the mid-price strategy on this dataset. After
filtering to one-tick local moves and removing sparse states, the learned correction is
small. This is a defensible empirical result, not necessarily an implementation error.

## Source-Code Provenance

The project strategy and backtester source code are original implementation files under
`src/cmf_backtester`. External repositories were used for research and sanity checks only:

- formulas and parameter interpretation;
- expected unit scale of inventory in the Avellaneda-Stoikov model;
- confirmation that many public examples are Monte Carlo simulations rather than
  historical LOB replay engines;
- confirmation that the microprice estimator should be fitted from local price moves.

No external repository source file was copied into the project.
