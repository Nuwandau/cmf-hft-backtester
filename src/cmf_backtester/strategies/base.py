from __future__ import annotations

from abc import ABC, abstractmethod

from cmf_backtester.execution.orders import Action
from cmf_backtester.market.snapshot import MarketSnapshot
from cmf_backtester.portfolio.portfolio import Portfolio


class BaseStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def on_market_update(
        self,
        snapshot: MarketSnapshot,
        portfolio: Portfolio,
        sigma_ticks_per_sqrt_second: float,
    ) -> list[Action]:
        raise NotImplementedError

    def reset(self) -> None:
        pass

