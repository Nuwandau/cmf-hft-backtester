import polars as pl

from cmf_backtester.liquidation.eda import queue_imbalance_next_move
from cmf_backtester.liquidation.features import (
    add_bbo_features,
    add_liquidation_features,
    add_ofi,
    add_trade_features,
)
from cmf_backtester.liquidation.io import timestamp_to_utc_string


def test_timestamp_microseconds_to_utc_string() -> None:
    assert timestamp_to_utc_string(1_700_000_000_000_000) == "2023-11-14T22:13:20+00:00"


def test_bbo_features_and_queue_imbalance() -> None:
    df = add_bbo_features(
        pl.DataFrame(
            {
                "bid_price": [99.0],
                "ask_price": [101.0],
                "bid_amount": [3.0],
                "ask_amount": [1.0],
            }
        )
    )
    row = df.row(0, named=True)
    assert row["mid"] == 100.0
    assert row["spread"] == 2.0
    assert row["spread_bps"] == 200.0
    assert row["queue_imbalance"] == 0.75


def test_trade_side_convention_and_clipped_weight() -> None:
    df = add_trade_features(
        pl.DataFrame(
            {
                "side": ["buy", "sell"],
                "price": [10_000.0, 20_000.0],
                "amount": [20.0, 1.0],
            }
        )
    )
    assert df["maker_direction"].to_list() == [1, -1]
    assert df["taker_direction"].to_list() == [1, -1]
    assert df["clipped_notional"].to_list() == [100_000.0, 20_000.0]


def test_bybit_liquidation_available_timestamp_shift() -> None:
    raw = pl.DataFrame(
        {"timestamp": [1_000_000], "side": ["buy"], "price": [100.0], "amount": [2.0]}
    )
    bybit = add_liquidation_features(raw, venue="bybit", bybit_delay_us=200_000)
    binance = add_liquidation_features(raw, venue="binance", bybit_delay_us=200_000)
    assert bybit["available_timestamp"].to_list() == [1_200_000]
    assert binance["available_timestamp"].to_list() == [1_000_000]
    assert bybit["signed_liquidation_notional"].to_list() == [200.0]


def test_ofi_uses_cont_kukanov_stoikov_l1_formula() -> None:
    df = add_ofi(
        pl.DataFrame(
            {
                "timestamp": [1, 2],
                "bid_price": [100.0, 101.0],
                "bid_amount": [5.0, 7.0],
                "ask_price": [102.0, 102.0],
                "ask_amount": [8.0, 6.0],
            }
        )
    )
    assert df["ofi"].to_list() == [0.0, 9.0]


def test_queue_imbalance_bucket_probability() -> None:
    sample = pl.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "bid_price": [100.0, 100.0, 101.0],
            "ask_price": [101.0, 101.0, 102.0],
            "bid_amount": [1.0, 9.0, 9.0],
            "ask_amount": [9.0, 1.0, 1.0],
        }
    )
    out = queue_imbalance_next_move(sample, "btcusdt", buckets=2)
    assert out["imbalance_bucket"].to_list() == [1, 2]
    assert out.filter(pl.col("imbalance_bucket") == 2)["prob_next_move_up"].item() == 1.0
