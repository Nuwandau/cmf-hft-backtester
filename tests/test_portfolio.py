from cmf_backtester.execution.orders import Fill, Side
from cmf_backtester.portfolio.portfolio import Portfolio


def test_portfolio_buy_sell_accounting() -> None:
    portfolio = Portfolio(tick_size=0.01)
    portfolio.apply_fill(Fill(1, Side.BUY, 100, 2.0, 1))
    assert portfolio.cash == -2.0
    assert portfolio.inventory == 2.0
    assert portfolio.turnover == 2.0
    portfolio.apply_fill(Fill(2, Side.SELL, 110, 1.0, 2))
    assert round(portfolio.cash, 12) == -0.9
    assert portfolio.inventory == 1.0
    assert portfolio.turnover == 3.1
    assert round(portfolio.mark_to_market(105), 12) == 0.15


def test_portfolio_fees_reduce_cash() -> None:
    portfolio = Portfolio(tick_size=0.01, fees_bps=10.0)
    portfolio.apply_fill(Fill(1, Side.BUY, 100, 1.0, 1))
    assert round(portfolio.cash, 6) == -1.001
