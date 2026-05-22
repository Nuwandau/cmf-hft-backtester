from __future__ import annotations

import numpy as np
import polars as pl


def add_bbo_features(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    return df.with_columns(
        [
            ((pl.col("bid_price") + pl.col("ask_price")) * 0.5).alias("mid"),
            (pl.col("ask_price") - pl.col("bid_price")).alias("spread"),
            (
                (pl.col("ask_price") - pl.col("bid_price"))
                / ((pl.col("bid_price") + pl.col("ask_price")) * 0.5)
                * 10_000
            ).alias("spread_bps"),
            (
                pl.col("bid_amount") / (pl.col("bid_amount") + pl.col("ask_amount"))
            ).alias("queue_imbalance"),
            (pl.col("bid_price") * pl.col("bid_amount")).alias("bid_notional"),
            (pl.col("ask_price") * pl.col("ask_amount")).alias("ask_notional"),
        ]
    )


def add_trade_features(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    return df.with_columns(
        [
            (pl.col("price") * pl.col("amount")).alias("notional"),
            (pl.col("price") * pl.col("amount")).clip(upper_bound=100_000).alias(
                "clipped_notional"
            ),
            (
                pl.when(pl.col("side") == "buy")
                .then(1)
                .when(pl.col("side") == "sell")
                .then(-1)
                .otherwise(None)
            ).alias("maker_direction"),
            (
                pl.when(pl.col("side") == "buy")
                .then(1)
                .when(pl.col("side") == "sell")
                .then(-1)
                .otherwise(None)
            ).alias("taker_direction"),
        ]
    )


def add_liquidation_features(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    venue: str,
    bybit_delay_us: int,
) -> pl.DataFrame | pl.LazyFrame:
    available = (
        pl.col("timestamp") + bybit_delay_us if venue == "bybit" else pl.col("timestamp")
    )
    return df.with_columns(
        [
            available.alias("available_timestamp"),
            (pl.col("price") * pl.col("amount")).alias("notional"),
            (pl.col("price") * pl.col("amount")).clip(upper_bound=100_000).alias(
                "clipped_notional"
            ),
            (
                pl.when(pl.col("side") == "buy")
                .then(1)
                .when(pl.col("side") == "sell")
                .then(-1)
                .otherwise(None)
            ).alias("liquidation_direction"),
        ]
    ).with_columns(
        (pl.col("liquidation_direction") * pl.col("notional")).alias(
            "signed_liquidation_notional"
        )
    )


def add_ofi(df: pl.DataFrame) -> pl.DataFrame:
    sorted_df = df.sort("timestamp")
    prev_bid_price = pl.col("bid_price").shift(1)
    prev_ask_price = pl.col("ask_price").shift(1)
    prev_bid_amount = pl.col("bid_amount").shift(1)
    prev_ask_amount = pl.col("ask_amount").shift(1)
    return sorted_df.with_columns(
        (
            pl.when(pl.col("bid_price") >= prev_bid_price)
            .then(pl.col("bid_amount"))
            .otherwise(0.0)
            - pl.when(pl.col("bid_price") <= prev_bid_price)
            .then(prev_bid_amount)
            .otherwise(0.0)
            - pl.when(pl.col("ask_price") <= prev_ask_price)
            .then(pl.col("ask_amount"))
            .otherwise(0.0)
            + pl.when(pl.col("ask_price") >= prev_ask_price)
            .then(prev_ask_amount)
            .otherwise(0.0)
        )
        .fill_null(0.0)
        .alias("ofi")
    )


def classify_trade_locations(df: pl.DataFrame, tolerance: float = 1e-12) -> pl.DataFrame:
    return df.with_columns(
        [
            (
                pl.when(pl.col("price") > pl.col("ask_price") + tolerance)
                .then(pl.lit("above_ask"))
                .when((pl.col("price") - pl.col("ask_price")).abs() <= tolerance)
                .then(pl.lit("at_ask"))
                .when(pl.col("price") < pl.col("bid_price") - tolerance)
                .then(pl.lit("below_bid"))
                .when((pl.col("price") - pl.col("bid_price")).abs() <= tolerance)
                .then(pl.lit("at_bid"))
                .when((pl.col("price") > pl.col("bid_price")) & (pl.col("price") < pl.col("ask_price")))
                .then(pl.lit("inside_spread"))
                .otherwise(pl.lit("outside_or_ambiguous"))
            ).alias("price_location")
        ]
    )


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return float("nan")
    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))
