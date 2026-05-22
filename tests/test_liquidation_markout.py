import polars as pl

from cmf_backtester.liquidation.features import add_bbo_features
from cmf_backtester.liquidation.markout import compute_markouts, join_trades_to_bbo


def test_backward_bbo_join_and_future_markout_signs() -> None:
    bbo = add_bbo_features(
        pl.DataFrame(
            {
                "timestamp": [0, 30_000_000],
                "bid_price": [99.0, 100.0],
                "bid_amount": [1.0, 1.0],
                "ask_price": [101.0, 102.0],
                "ask_amount": [1.0, 1.0],
            }
        )
    )
    trades = pl.DataFrame(
        {
            "timestamp": [1_000_000, 1_000_000],
            "ticker": ["perp:btcusdt", "perp:btcusdt"],
            "symbol": ["btcusdt", "btcusdt"],
            "side": ["buy", "sell"],
            "price": [100.0, 100.0],
            "amount": [1.0, 1.0],
        }
    )

    joined = join_trades_to_bbo(trades, bbo, tolerance_us=5_000_000)
    assert joined["bbo_timestamp"].to_list() == [0, 0]
    assert joined["bbo_age_us"].to_list() == [1_000_000, 1_000_000]

    markouts = compute_markouts(
        joined,
        bbo,
        horizons_seconds=(30,),
        maker_rebate_bps=0.5,
        tolerance_us=5_000_000,
    ).sort("side")

    buy_row = markouts.filter(pl.col("side") == "buy").row(0, named=True)
    sell_row = markouts.filter(pl.col("side") == "sell").row(0, named=True)
    assert buy_row["future_mid_30s"] == 101.0
    assert buy_row["future_bbo_age_us_30s"] == 1_000_000
    assert buy_row["pnl_bps_30s"] == -99.5
    assert sell_row["pnl_bps_30s"] == 100.5


def test_stale_future_bbo_is_excluded() -> None:
    bbo = add_bbo_features(
        pl.DataFrame(
            {
                "timestamp": [0],
                "bid_price": [99.0],
                "bid_amount": [1.0],
                "ask_price": [101.0],
                "ask_amount": [1.0],
            }
        )
    )
    trades = pl.DataFrame(
        {
            "timestamp": [1_000_000],
            "ticker": ["perp:btcusdt"],
            "symbol": ["btcusdt"],
            "side": ["buy"],
            "price": [100.0],
            "amount": [1.0],
        }
    )
    joined = join_trades_to_bbo(trades, bbo, tolerance_us=5_000_000)
    markouts = compute_markouts(
        joined,
        bbo,
        horizons_seconds=(30,),
        maker_rebate_bps=0.5,
        tolerance_us=5_000_000,
    )
    assert markouts["future_mid_30s"].to_list() == [None]
    assert markouts["pnl_bps_30s"].to_list() == [None]
