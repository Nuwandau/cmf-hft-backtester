import numpy as np

from cmf_backtester.backtest.engine import BacktestEngine
from cmf_backtester.data.loaders import MarketDataArrays
from cmf_backtester.execution.orders import PlaceOrder, Side
from cmf_backtester.market.snapshot import MarketSnapshot
from cmf_backtester.portfolio.portfolio import Portfolio
from cmf_backtester.strategies.avellaneda_stoikov import (
    AvellanedaStoikovConfig,
    AvellanedaStoikovStrategy,
)


def tiny_arrays() -> MarketDataArrays:
    return MarketDataArrays(
        timestamps=np.array([1, 2, 3, 4], dtype=np.int64),
        best_bid_ticks=np.array([100, 100, 99, 101], dtype=np.int64),
        best_ask_ticks=np.array([101, 101, 100, 102], dtype=np.int64),
        bid_size=np.ones(4),
        ask_size=np.ones(4),
        mid_ticks=np.array([100.5, 100.5, 99.5, 101.5]),
        mid_half_ticks=np.array([201, 201, 199, 203], dtype=np.int64),
        spread_ticks=np.ones(4, dtype=np.int64),
        imbalance=np.full(4, 0.5),
        split=np.array(["test"] * 4),
        date=np.array(["2024-01-01"] * 4),
        tick_size=0.01,
    )


def test_debug_and_numba_backtest_match() -> None:
    arrays = tiny_arrays()
    cfg = AvellanedaStoikovConfig(
        gamma=1e-6,
        k=0.5,
        tau_seconds=1.0,
        order_size=1.0,
        max_inventory=10.0,
        min_spread_ticks=1,
        quote_refresh_seconds=0.0,
        post_only=True,
    )
    sigma = np.full(len(arrays), 0.1)
    debug = BacktestEngine(
        arrays, AvellanedaStoikovStrategy(cfg), sigma, runtime_mode="debug_python"
    ).run()
    fast = BacktestEngine(
        arrays, AvellanedaStoikovStrategy(cfg), sigma, runtime_mode="fast_numba"
    ).run()
    assert debug.pnl[-1] == fast.pnl[-1]
    assert debug.inventory[-1] == fast.inventory[-1]
    assert debug.fill_count[-1] == fast.fill_count[-1]


def test_as_inventory_risk_unit_uses_lot_inventory() -> None:
    cfg = AvellanedaStoikovConfig(
        gamma=1e-4,
        k=0.25,
        tau_seconds=180.0,
        order_size=10_000.0,
        max_inventory=50_000.0,
        inventory_risk_unit=10_000.0,
        min_spread_ticks=1,
        quote_refresh_seconds=1.0,
        post_only=False,
    )
    strategy = AvellanedaStoikovStrategy(cfg)
    snapshot = MarketSnapshot(
        timestamp=1,
        best_bid_ticks=999,
        best_ask_ticks=1001,
        bid_size=10_000.0,
        ask_size=10_000.0,
        mid_ticks=1000.0,
        mid_half_ticks=2000,
        spread_ticks=2,
        imbalance=0.5,
        tick_size=0.01,
    )
    portfolio = Portfolio(tick_size=0.01, inventory=10_000.0)

    actions = strategy.on_market_update(snapshot, portfolio, sigma_ticks_per_sqrt_second=20.0)
    orders = [action for action in actions if isinstance(action, PlaceOrder)]
    bid = next(order for order in orders if order.side == Side.BUY)
    ask = next(order for order in orders if order.side == Side.SELL)

    assert 900 < bid.price_ticks < 1000
    assert 990 < ask.price_ticks < 1100


def test_debug_and_numba_partial_fill_match() -> None:
    arrays = tiny_arrays()
    cfg = AvellanedaStoikovConfig(
        gamma=1e-6,
        k=0.5,
        tau_seconds=1.0,
        order_size=5.0,
        max_inventory=10.0,
        min_spread_ticks=1,
        quote_refresh_seconds=10.0,
        post_only=True,
    )
    sigma = np.full(len(arrays), 0.1)
    debug = BacktestEngine(
        arrays,
        AvellanedaStoikovStrategy(cfg),
        sigma,
        runtime_mode="debug_python",
        fill_mode="visible_size",
    ).run()
    fast = BacktestEngine(
        arrays,
        AvellanedaStoikovStrategy(cfg),
        sigma,
        runtime_mode="fast_numba",
        fill_mode="visible_size",
    ).run()
    assert debug.pnl[-1] == fast.pnl[-1]
    assert debug.inventory[-1] == fast.inventory[-1]
    assert debug.fill_count[-1] == fast.fill_count[-1]
