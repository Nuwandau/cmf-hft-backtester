from __future__ import annotations

import math
from dataclasses import dataclass

from cmf_backtester.execution.orders import Action, CancelAll, PlaceOrder, Side
from cmf_backtester.market.snapshot import MarketSnapshot
from cmf_backtester.portfolio.portfolio import Portfolio
from cmf_backtester.strategies.base import BaseStrategy


@dataclass(frozen=True)
class AvellanedaStoikovConfig:
    gamma: float
    k: float
    tau_seconds: float
    order_size: float
    max_inventory: float
    inventory_risk_unit: float | None = None
    min_spread_ticks: int = 1
    quote_refresh_seconds: float = 1.0
    post_only: bool = True


class AvellanedaStoikovStrategy(BaseStrategy):
    def __init__(self, config: AvellanedaStoikovConfig) -> None:
        self.config = config
        self.last_quote_timestamp: int | None = None

    @property
    def name(self) -> str:
        return "avellaneda_stoikov_mid"

    def reference_price_ticks(self, snapshot: MarketSnapshot) -> float:
        return snapshot.mid_ticks

    def should_refresh(self, timestamp: int) -> bool:
        if self.last_quote_timestamp is None:
            return True
        refresh_us = int(round(self.config.quote_refresh_seconds * 1_000_000))
        return timestamp - self.last_quote_timestamp >= refresh_us

    def on_market_update(
        self,
        snapshot: MarketSnapshot,
        portfolio: Portfolio,
        sigma_ticks_per_sqrt_second: float,
    ) -> list[Action]:
        if not self.should_refresh(snapshot.timestamp):
            return []

        cfg = self.config
        gamma = max(float(cfg.gamma), 1e-18)
        k = max(float(cfg.k), 1e-18)
        variance_horizon = sigma_ticks_per_sqrt_second**2 * cfg.tau_seconds
        reference_ticks = self.reference_price_ticks(snapshot)
        inventory_risk_unit = float(cfg.inventory_risk_unit or cfg.order_size)
        inventory_risk_unit = max(inventory_risk_unit, 1e-18)
        model_inventory = portfolio.inventory / inventory_risk_unit
        reservation_ticks = reference_ticks - model_inventory * gamma * variance_horizon
        total_spread_ticks = gamma * variance_horizon + (2.0 / gamma) * math.log1p(gamma / k)
        total_spread_ticks = max(float(cfg.min_spread_ticks), total_spread_ticks)

        raw_bid = reservation_ticks - total_spread_ticks / 2.0
        raw_ask = reservation_ticks + total_spread_ticks / 2.0
        bid_ticks = int(math.floor(raw_bid))
        ask_ticks = int(math.ceil(raw_ask))

        if ask_ticks - bid_ticks < cfg.min_spread_ticks:
            ask_ticks = bid_ticks + cfg.min_spread_ticks

        if cfg.post_only:
            bid_ticks = min(bid_ticks, snapshot.best_bid_ticks)
            ask_ticks = max(ask_ticks, snapshot.best_ask_ticks)
            if ask_ticks - bid_ticks < cfg.min_spread_ticks:
                ask_ticks = bid_ticks + cfg.min_spread_ticks

        actions: list[Action] = [CancelAll()]
        if portfolio.inventory < cfg.max_inventory:
            actions.append(PlaceOrder(Side.BUY, bid_ticks, cfg.order_size))
        if portfolio.inventory > -cfg.max_inventory:
            actions.append(PlaceOrder(Side.SELL, ask_ticks, cfg.order_size))
        self.last_quote_timestamp = snapshot.timestamp
        return actions

    def reset(self) -> None:
        self.last_quote_timestamp = None


def config_from_dict(raw: dict) -> AvellanedaStoikovConfig:
    return AvellanedaStoikovConfig(
        gamma=float(raw["gamma"]),
        k=float(raw["k"]),
        tau_seconds=float(raw["tau_seconds"]),
        order_size=float(raw["order_size"]),
        max_inventory=float(raw["max_inventory"]),
        inventory_risk_unit=(
            float(raw["inventory_risk_unit"]) if raw.get("inventory_risk_unit") is not None else None
        ),
        min_spread_ticks=int(raw.get("min_spread_ticks", 1)),
        quote_refresh_seconds=float(raw.get("quote_refresh_seconds", 1.0)),
        post_only=bool(raw.get("post_only", True)),
    )
