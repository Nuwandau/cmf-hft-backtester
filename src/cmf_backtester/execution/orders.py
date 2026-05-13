from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    ACTIVE = "active"
    FILLED = "filled"
    CANCELLED = "cancelled"


@dataclass
class Order:
    order_id: int
    side: Side
    price_ticks: int
    quantity: float
    remaining_quantity: float
    status: OrderStatus
    timestamp_created: int
    timestamp_closed: int | None = None


@dataclass(frozen=True)
class Fill:
    order_id: int
    side: Side
    price_ticks: int
    quantity: float
    timestamp: int


@dataclass(frozen=True)
class PlaceOrder:
    side: Side
    price_ticks: int
    quantity: float


@dataclass(frozen=True)
class CancelAll:
    pass


Action = PlaceOrder | CancelAll

