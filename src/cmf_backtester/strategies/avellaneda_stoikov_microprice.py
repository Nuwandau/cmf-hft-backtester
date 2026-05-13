from __future__ import annotations

from cmf_backtester.market.snapshot import MarketSnapshot
from cmf_backtester.strategies.avellaneda_stoikov import (
    AvellanedaStoikovConfig,
    AvellanedaStoikovStrategy,
)


class AvellanedaStoikovMicropriceStrategy(AvellanedaStoikovStrategy):
    def __init__(self, config: AvellanedaStoikovConfig) -> None:
        super().__init__(config)

    @property
    def name(self) -> str:
        return "avellaneda_stoikov_microprice"

    def reference_price_ticks(self, snapshot: MarketSnapshot) -> float:
        return snapshot.microprice_ticks

