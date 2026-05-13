from __future__ import annotations

from dataclasses import dataclass, field

from cmf_backtester.execution.orders import Fill, Side


@dataclass
class Portfolio:
    tick_size: float
    fees_bps: float = 0.0
    cash: float = 0.0
    inventory: float = 0.0
    turnover: float = 0.0
    fills: list[Fill] = field(default_factory=list)
    buy_fills: int = 0
    sell_fills: int = 0

    def apply_fill(self, fill: Fill) -> None:
        price = fill.price_ticks * self.tick_size
        value = price * fill.quantity
        fee = abs(value) * self.fees_bps / 10_000.0
        if fill.side == Side.BUY:
            self.cash -= value + fee
            self.inventory += fill.quantity
            self.buy_fills += 1
        else:
            self.cash += value - fee
            self.inventory -= fill.quantity
            self.sell_fills += 1
        self.turnover += abs(value)
        self.fills.append(fill)

    def mark_to_market(self, mid_ticks: float) -> float:
        return self.cash + self.inventory * mid_ticks * self.tick_size

    @property
    def fill_count(self) -> int:
        return len(self.fills)
