from __future__ import annotations

import math

import numpy as np

from cmf_backtester.backtest.kernels import run_as_crossing_kernel
from cmf_backtester.backtest.recorder import BacktestResult
from cmf_backtester.data.loaders import MarketDataArrays
from cmf_backtester.execution.execution_model import CrossingExecutionModel
from cmf_backtester.execution.orders import CancelAll, Order, OrderStatus, PlaceOrder
from cmf_backtester.market.snapshot import MarketSnapshot
from cmf_backtester.portfolio.portfolio import Portfolio
from cmf_backtester.strategies.avellaneda_stoikov import AvellanedaStoikovStrategy
from cmf_backtester.strategies.base import BaseStrategy


class BacktestEngine:
    def __init__(
        self,
        market_data: MarketDataArrays,
        strategy: BaseStrategy,
        sigma_ticks_per_sqrt_second: np.ndarray,
        microprice_adjustment_ticks: np.ndarray | None = None,
        runtime_mode: str = "debug_python",
        fill_mode: str = "full",
        fees_bps: float = 0.0,
    ) -> None:
        if len(market_data) != len(sigma_ticks_per_sqrt_second):
            raise ValueError("sigma array length must match market data")
        self.market_data = market_data
        self.strategy = strategy
        self.sigma = sigma_ticks_per_sqrt_second.astype(np.float64)
        if microprice_adjustment_ticks is None:
            microprice_adjustment_ticks = np.zeros(len(market_data), dtype=np.float64)
        if len(market_data) != len(microprice_adjustment_ticks):
            raise ValueError("microprice adjustment length must match market data")
        self.microprice_adjustment_ticks = microprice_adjustment_ticks.astype(np.float64)
        self.execution_model = CrossingExecutionModel(fill_mode)
        self.runtime_mode = runtime_mode
        self.fill_mode = fill_mode
        self.fees_bps = float(fees_bps)

    def run(self) -> BacktestResult:
        if self.runtime_mode == "fast_numba":
            return self._run_fast_numba()
        return self._run_debug_python()

    def _run_fast_numba(self) -> BacktestResult:
        if not isinstance(self.strategy, AvellanedaStoikovStrategy):
            raise TypeError("fast_numba mode currently supports Avellaneda-Stoikov strategies only")
        data = self.market_data
        cfg = self.strategy.config
        (
            pnl,
            cash,
            inventory,
            turnover,
            fill_count,
            buy_fills,
            sell_fills,
            reservation_ticks,
            bid_quote_ticks,
            ask_quote_ticks,
        ) = run_as_crossing_kernel(
            data.timestamps.astype(np.int64),
            data.best_bid_ticks.astype(np.int64),
            data.best_ask_ticks.astype(np.int64),
            data.bid_size.astype(np.float64),
            data.ask_size.astype(np.float64),
            data.mid_ticks.astype(np.float64),
            self.sigma.astype(np.float64),
            self.microprice_adjustment_ticks.astype(np.float64),
            float(data.tick_size),
            float(cfg.gamma),
            float(cfg.k),
            float(cfg.tau_seconds),
            float(cfg.order_size),
            float(cfg.max_inventory),
            float(cfg.inventory_risk_unit or cfg.order_size),
            int(cfg.min_spread_ticks),
            float(cfg.quote_refresh_seconds),
            bool(cfg.post_only),
            1 if self.fill_mode == "visible_size" else 0,
            float(self.fees_bps),
        )
        return BacktestResult(
            strategy_name=self.strategy.name,
            timestamps=data.timestamps.copy(),
            mid_ticks=data.mid_ticks.copy(),
            microprice_ticks=data.mid_ticks + self.microprice_adjustment_ticks,
            pnl=pnl,
            cash=cash,
            inventory=inventory,
            turnover=turnover,
            fill_count=fill_count,
            buy_fills=buy_fills,
            sell_fills=sell_fills,
            reservation_ticks=reservation_ticks,
            bid_quote_ticks=bid_quote_ticks,
            ask_quote_ticks=ask_quote_ticks,
            tick_size=data.tick_size,
        )

    def _run_debug_python(self) -> BacktestResult:
        data = self.market_data
        n = len(data)
        portfolio = Portfolio(tick_size=data.tick_size, fees_bps=self.fees_bps)
        active_orders: list[Order] = []
        next_order_id = 1
        self.strategy.reset()

        pnl = np.zeros(n, dtype=np.float64)
        cash = np.zeros(n, dtype=np.float64)
        inventory = np.zeros(n, dtype=np.float64)
        turnover = np.zeros(n, dtype=np.float64)
        fill_count = np.zeros(n, dtype=np.int64)
        buy_fills = np.zeros(n, dtype=np.int64)
        sell_fills = np.zeros(n, dtype=np.int64)
        reservation_ticks = np.full(n, np.nan, dtype=np.float64)
        bid_quote_ticks = np.full(n, np.nan, dtype=np.float64)
        ask_quote_ticks = np.full(n, np.nan, dtype=np.float64)
        current_bid_quote = math.nan
        current_ask_quote = math.nan
        current_reservation = math.nan

        for i in range(n):
            snapshot = MarketSnapshot(
                timestamp=int(data.timestamps[i]),
                best_bid_ticks=int(data.best_bid_ticks[i]),
                best_ask_ticks=int(data.best_ask_ticks[i]),
                bid_size=float(data.bid_size[i]),
                ask_size=float(data.ask_size[i]),
                mid_ticks=float(data.mid_ticks[i]),
                mid_half_ticks=int(data.mid_half_ticks[i]),
                spread_ticks=int(data.spread_ticks[i]),
                imbalance=float(data.imbalance[i]),
                tick_size=data.tick_size,
                microprice_adjustment_ticks=float(self.microprice_adjustment_ticks[i]),
            )

            fills = self.execution_model.match(active_orders, snapshot)
            for fill in fills:
                portfolio.apply_fill(fill)
            active_orders = [o for o in active_orders if o.status == OrderStatus.ACTIVE]

            actions = self.strategy.on_market_update(snapshot, portfolio, float(self.sigma[i]))
            for action in actions:
                if isinstance(action, CancelAll):
                    for order in active_orders:
                        order.status = OrderStatus.CANCELLED
                        order.timestamp_closed = snapshot.timestamp
                    active_orders.clear()
                    current_bid_quote = math.nan
                    current_ask_quote = math.nan
                    current_reservation = math.nan
                elif isinstance(action, PlaceOrder):
                    order = Order(
                        order_id=next_order_id,
                        side=action.side,
                        price_ticks=int(action.price_ticks),
                        quantity=float(action.quantity),
                        remaining_quantity=float(action.quantity),
                        status=OrderStatus.ACTIVE,
                        timestamp_created=snapshot.timestamp,
                    )
                    next_order_id += 1
                    active_orders.append(order)
                    if action.side.value == "buy":
                        current_bid_quote = float(action.price_ticks)
                    else:
                        current_ask_quote = float(action.price_ticks)
            if isinstance(self.strategy, AvellanedaStoikovStrategy):
                cfg = self.strategy.config
                gamma = max(float(cfg.gamma), 1e-18)
                inventory_risk_unit = max(float(cfg.inventory_risk_unit or cfg.order_size), 1e-18)
                variance_horizon = float(self.sigma[i]) ** 2 * cfg.tau_seconds
                reference_ticks = snapshot.mid_ticks + snapshot.microprice_adjustment_ticks
                current_reservation = (
                    reference_ticks
                    - (portfolio.inventory / inventory_risk_unit) * gamma * variance_horizon
                )

            pnl[i] = portfolio.mark_to_market(snapshot.mid_ticks)
            cash[i] = portfolio.cash
            inventory[i] = portfolio.inventory
            turnover[i] = portfolio.turnover
            fill_count[i] = portfolio.fill_count
            buy_fills[i] = portfolio.buy_fills
            sell_fills[i] = portfolio.sell_fills
            reservation_ticks[i] = current_reservation
            bid_quote_ticks[i] = current_bid_quote
            ask_quote_ticks[i] = current_ask_quote

        return BacktestResult(
            strategy_name=self.strategy.name,
            timestamps=data.timestamps.copy(),
            mid_ticks=data.mid_ticks.copy(),
            microprice_ticks=data.mid_ticks + self.microprice_adjustment_ticks,
            pnl=pnl,
            cash=cash,
            inventory=inventory,
            turnover=turnover,
            fill_count=fill_count,
            buy_fills=buy_fills,
            sell_fills=sell_fills,
            reservation_ticks=reservation_ticks,
            bid_quote_ticks=bid_quote_ticks,
            ask_quote_ticks=ask_quote_ticks,
            tick_size=data.tick_size,
        )
