from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketSnapshot:
    timestamp: int
    best_bid_ticks: int
    best_ask_ticks: int
    bid_size: float
    ask_size: float
    mid_ticks: float
    mid_half_ticks: int
    spread_ticks: int
    imbalance: float
    tick_size: float
    microprice_adjustment_ticks: float = 0.0

    @property
    def mid_price(self) -> float:
        return self.mid_ticks * self.tick_size

    @property
    def microprice_ticks(self) -> float:
        return self.mid_ticks + self.microprice_adjustment_ticks

    @property
    def microprice(self) -> float:
        return self.microprice_ticks * self.tick_size

