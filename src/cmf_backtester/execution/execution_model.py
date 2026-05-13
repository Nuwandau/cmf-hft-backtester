from __future__ import annotations

from cmf_backtester.execution.orders import Fill, Order, OrderStatus, Side
from cmf_backtester.market.snapshot import MarketSnapshot


class CrossingExecutionModel:
    """Crossing model based on best bid/ask snapshots.

    `fill_mode="full"` fills the whole remaining order once crossed.
    `fill_mode="visible_size"` fills at most the visible top-of-book quantity. This is a
    simple partial-fill approximation, not a queue-position model.
    """

    def __init__(self, fill_mode: str = "full") -> None:
        if fill_mode not in {"full", "visible_size"}:
            raise ValueError(f"Unsupported fill_mode: {fill_mode}")
        self.fill_mode = fill_mode

    def match(self, orders: list[Order], snapshot: MarketSnapshot) -> list[Fill]:
        fills: list[Fill] = []
        for order in orders:
            if order.status != OrderStatus.ACTIVE:
                continue
            if order.timestamp_created >= snapshot.timestamp:
                continue
            if order.side == Side.BUY:
                crossed = snapshot.best_ask_ticks <= order.price_ticks
            else:
                crossed = snapshot.best_bid_ticks >= order.price_ticks
            if crossed:
                if self.fill_mode == "visible_size":
                    visible_size = snapshot.ask_size if order.side == Side.BUY else snapshot.bid_size
                    fill_qty = min(order.remaining_quantity, max(0.0, visible_size))
                else:
                    fill_qty = order.remaining_quantity
                if fill_qty <= 0:
                    continue
                fills.append(
                    Fill(
                        order_id=order.order_id,
                        side=order.side,
                        price_ticks=order.price_ticks,
                        quantity=fill_qty,
                        timestamp=snapshot.timestamp,
                    )
                )
                order.remaining_quantity -= fill_qty
                if order.remaining_quantity <= 1e-12:
                    order.remaining_quantity = 0.0
                    order.status = OrderStatus.FILLED
                    order.timestamp_closed = snapshot.timestamp
        return fills
