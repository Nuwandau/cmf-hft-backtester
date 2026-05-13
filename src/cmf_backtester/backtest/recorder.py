from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl


@dataclass(frozen=True)
class BacktestResult:
    strategy_name: str
    timestamps: np.ndarray
    mid_ticks: np.ndarray
    microprice_ticks: np.ndarray
    pnl: np.ndarray
    cash: np.ndarray
    inventory: np.ndarray
    turnover: np.ndarray
    fill_count: np.ndarray
    buy_fills: np.ndarray
    sell_fills: np.ndarray
    reservation_ticks: np.ndarray
    bid_quote_ticks: np.ndarray
    ask_quote_ticks: np.ndarray
    tick_size: float

    def to_frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "timestamp": self.timestamps,
                "mid_price": self.mid_ticks * self.tick_size,
                "microprice": self.microprice_ticks * self.tick_size,
                "pnl": self.pnl,
                "cash": self.cash,
                "inventory": self.inventory,
                "turnover": self.turnover,
                "fill_count": self.fill_count,
                "buy_fills": self.buy_fills,
                "sell_fills": self.sell_fills,
                "reservation_price": self.reservation_ticks * self.tick_size,
                "bid_quote": self.bid_quote_ticks * self.tick_size,
                "ask_quote": self.ask_quote_ticks * self.tick_size,
            }
        )

    def write_parquet(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().write_parquet(path)
