from cmf_backtester.execution.execution_model import CrossingExecutionModel
from cmf_backtester.execution.orders import Order, OrderStatus, Side
from cmf_backtester.market.snapshot import MarketSnapshot


def snapshot(ts: int, bid: int, ask: int, bid_size: float = 1.0, ask_size: float = 1.0) -> MarketSnapshot:
    return MarketSnapshot(
        timestamp=ts,
        best_bid_ticks=bid,
        best_ask_ticks=ask,
        bid_size=bid_size,
        ask_size=ask_size,
        mid_ticks=(bid + ask) / 2,
        mid_half_ticks=bid + ask,
        spread_ticks=ask - bid,
        imbalance=0.5,
        tick_size=0.01,
    )


def test_buy_crossing_fill_future_only() -> None:
    model = CrossingExecutionModel()
    order = Order(1, Side.BUY, 100, 1.0, 1.0, OrderStatus.ACTIVE, timestamp_created=10)
    assert model.match([order], snapshot(10, 99, 100)) == []
    fills = model.match([order], snapshot(11, 99, 100))
    assert len(fills) == 1
    assert order.status == OrderStatus.FILLED


def test_sell_crossing_fill() -> None:
    model = CrossingExecutionModel()
    order = Order(1, Side.SELL, 101, 1.0, 1.0, OrderStatus.ACTIVE, timestamp_created=10)
    fills = model.match([order], snapshot(11, 101, 102))
    assert len(fills) == 1


def test_visible_size_partial_fill_keeps_order_active() -> None:
    model = CrossingExecutionModel(fill_mode="visible_size")
    order = Order(1, Side.BUY, 100, 10.0, 10.0, OrderStatus.ACTIVE, timestamp_created=10)
    snap = snapshot(11, 99, 100, ask_size=3.0)
    fills = model.match([order], snap)
    assert len(fills) == 1
    assert fills[0].quantity == 3.0
    assert order.remaining_quantity == 7.0
    assert order.status == OrderStatus.ACTIVE
