# Papers Relevance Audit For Liquidation EDA

This note summarizes which local papers are useful for the liquidation EDA and
future signal work.

## Directly Relevant Now

- `Fragmentation,_Price_Formation,_and_Cross_Impact_in_Bitcoin_Markets.pdf`:
  crypto venue fragmentation, short-horizon features, trade-flow imbalance,
  lead-lag, cross-impact, adverse-selection curves.
- `Order flow and price formation.pdf`:
  order-flow persistence, response functions, market impact, cross-impact,
  confounding and heavy-tail concerns.
- `The_Price_Impact_of_Order_Book_Events_—_Rama_Cont,_Arseniy_Kukanov.pdf`:
  L1 order flow imbalance. We can compute an OFI-style feature from Binance
  bookticker updates.
- `Queue_Imbalance_as_a_One_Tick_Ahead_Price_Predictor_in_a_Limit_Order.pdf`:
  queue imbalance buckets and one-tick mid-move probabilities.

## Useful For Future Signal / ML Work

- `DeepLOB_Deep_Convolutional_Neural_Networks_for_Limit_Order_Books.pdf`:
  future deep learning reference, especially normalization and non-stationarity.
  Not needed for first EDA.
- `David_Easley,_Marcos_López_de_Prado,_Maureen_O’Hara_—_The_Volume.pdf`:
  VPIN/order-flow toxicity intuition. Could inspire volume-bucketed toxicity
  diagnostics, but should not dominate the first implementation.
- `Hasbrouck's book copy.pdf`:
  broad empirical microstructure reference for trade classification, effective
  spread, and response functions.

## Broad Background / Roadmap

- `Barry Johnson - Algorithmic Trading and DMA...pdf`:
  market access, order types, execution mechanics.
- `Olivier_Guéant_The_Financial_Mathematics_of_Market_Liquidity_From.pdf`:
  optimal execution and market making background.
- `The science of algorithmic trading and portfolio management...pdf`:
  execution cost and portfolio trading background.
- `High_Frequency_Trading_Small-Cap_ru copy.pdf` and
  `Inside_the_Black_Box_Small-Cap_ru copy.pdf`:
  general HFT/quant trading background.
- `Option-Volatility-and-Pricing-pdf-free-download copy.pdf`:
  not directly relevant to liquidation EDA.

## EDA Additions Motivated By Literature

- Add subsecond signed trade-flow windows: `100ms`, `250ms`, `500ms`, `1s`, `2s`.
- Add BBO-derived OFI features from bookticker updates.
- Add queue-imbalance buckets and next-mid-move probabilities.
- Add signed-flow response functions for trades and liquidations.
- Add nonlinear bucket plots and signed sqrt/log transformations for notional flow.
- Add adverse-selection curves over many horizons.
- Add cross-asset BTC/ETH stress diagnostics.
