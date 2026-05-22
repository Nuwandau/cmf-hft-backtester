from __future__ import annotations

import math
from datetime import datetime, timezone

import polars as pl

from cmf_backtester.liquidation.features import add_bbo_features, add_liquidation_features
from cmf_backtester.liquidation.io import collect_frame, date_expr, hourly_event_counts
from cmf_backtester.liquidation.schema import SourceSpec


def split_expr() -> pl.Expr:
    date = date_expr()
    return (
        pl.when((date >= "2025-12-01") & (date <= "2026-01-31"))
        .then(pl.lit("train"))
        .when((date >= "2026-02-01") & (date <= "2026-02-28"))
        .then(pl.lit("validation"))
        .otherwise(pl.lit("outside"))
        .alias("split")
    )


def add_split_column(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="us").dt.strftime("%Y-%m-%d").alias("date")
    ).with_columns(
        (
            pl.when((pl.col("date") >= "2025-12-01") & (pl.col("date") <= "2026-01-31"))
            .then(pl.lit("train"))
            .when((pl.col("date") >= "2026-02-01") & (pl.col("date") <= "2026-02-28"))
            .then(pl.lit("validation"))
            .otherwise(pl.lit("outside"))
        ).alias("split")
    )


def bbo_quality(sample: pl.DataFrame, symbol: str) -> pl.DataFrame:
    df = add_bbo_features(sample)
    return df.select(
        pl.lit(symbol).alias("symbol"),
        pl.len().alias("sample_rows"),
        (pl.col("bid_price") <= 0).sum().alias("nonpositive_bid_rows"),
        (pl.col("ask_price") <= 0).sum().alias("nonpositive_ask_rows"),
        (pl.col("bid_amount") < 0).sum().alias("negative_bid_amount_rows"),
        (pl.col("ask_amount") < 0).sum().alias("negative_ask_amount_rows"),
        (pl.col("bid_price") > pl.col("ask_price")).sum().alias("crossed_rows"),
        (pl.col("bid_price") == pl.col("ask_price")).sum().alias("locked_rows"),
        pl.col("spread_bps").mean().alias("mean_spread_bps"),
        pl.col("spread_bps").median().alias("median_spread_bps"),
        pl.col("spread_bps").quantile(0.99).alias("p99_spread_bps"),
        pl.col("queue_imbalance").mean().alias("mean_queue_imbalance"),
    )


def bbo_ofi_summary(sample_with_ofi: pl.DataFrame, symbol: str) -> pl.DataFrame:
    df = sample_with_ofi
    return df.select(
        pl.lit(symbol).alias("symbol"),
        pl.len().alias("sample_rows"),
        pl.col("ofi").mean().alias("mean_ofi"),
        pl.col("ofi").median().alias("median_ofi"),
        pl.col("ofi").quantile(0.01).alias("p01_ofi"),
        pl.col("ofi").quantile(0.99).alias("p99_ofi"),
        pl.corr("ofi", "next_return_bps").alias("corr_ofi_next_return_bps"),
    )


def queue_imbalance_next_move(sample: pl.DataFrame, symbol: str, buckets: int = 10) -> pl.DataFrame:
    df = add_bbo_features(sample).sort("timestamp").with_columns(
        [
            pl.col("mid").shift(-1).alias("next_mid"),
            (
                (pl.col("queue_imbalance") * buckets)
                .floor()
                .clip(upper_bound=buckets - 1)
                .cast(pl.Int64)
                + 1
            ).alias("imbalance_bucket"),
        ]
    ).with_columns(
        [
            (pl.col("next_mid") > pl.col("mid")).alias("next_move_up"),
            (pl.col("next_mid") < pl.col("mid")).alias("next_move_down"),
        ]
    )
    return (
        df.group_by("imbalance_bucket")
        .agg(
            pl.lit(symbol).first().alias("symbol"),
            pl.len().alias("rows"),
            pl.col("queue_imbalance").mean().alias("mean_queue_imbalance"),
            pl.col("next_move_up").mean().alias("prob_next_move_up"),
            pl.col("next_move_down").mean().alias("prob_next_move_down"),
        )
        .sort("imbalance_bucket")
    )


def trade_summary(sample: pl.DataFrame, symbol: str) -> pl.DataFrame:
    df = sample.with_columns(
        [
            (pl.col("price") * pl.col("amount")).alias("notional"),
            (pl.col("price") * pl.col("amount")).clip(upper_bound=100_000).alias(
                "clipped_notional"
            ),
        ]
    )
    return (
        df.group_by("side")
        .agg(
            pl.lit(symbol).first().alias("symbol"),
            pl.len().alias("sample_rows"),
            pl.col("amount").mean().alias("mean_amount"),
            pl.col("amount").median().alias("median_amount"),
            pl.col("notional").mean().alias("mean_notional"),
            pl.col("notional").median().alias("median_notional"),
            pl.col("notional").quantile(0.99).alias("p99_notional"),
            pl.col("clipped_notional").sum().alias("sample_clipped_turnover"),
        )
        .sort("side")
    )


def liquidation_summary(sample: pl.DataFrame, symbol: str, venue: str, bybit_delay_us: int) -> pl.DataFrame:
    df = add_liquidation_features(sample, venue=venue, bybit_delay_us=bybit_delay_us)
    return (
        df.group_by("side")
        .agg(
            pl.lit(symbol).first().alias("symbol"),
            pl.lit(venue).first().alias("venue"),
            pl.len().alias("sample_rows"),
            pl.col("amount").mean().alias("mean_amount"),
            pl.col("amount").median().alias("median_amount"),
            pl.col("notional").mean().alias("mean_notional"),
            pl.col("notional").median().alias("median_notional"),
            pl.col("notional").quantile(0.99).alias("p99_notional"),
            pl.col("notional").sum().alias("sample_notional"),
        )
        .sort("side")
    )


def price_location_summary(joined: pl.DataFrame, symbol: str) -> pl.DataFrame:
    groups = ["side", "price_location"]
    if "split" in joined.columns:
        groups = ["split", *groups]
    return (
        joined.group_by(groups)
        .agg(pl.lit(symbol).first().alias("symbol"), pl.len().alias("rows"))
        .with_columns(
            (pl.col("rows") / pl.col("rows").sum().over([col for col in groups if col != "price_location"])).alias(
                "share_within_side"
            )
        )
        .sort(groups)
    )


def convention_examples(joined: pl.DataFrame, horizons_seconds: tuple[int, ...]) -> pl.DataFrame:
    cols = [
        "symbol",
        "timestamp",
        "side",
        "price",
        "bid_price",
        "ask_price",
        "price_location",
        "maker_direction",
        "clipped_notional",
    ]
    for horizon in horizons_seconds[:1]:
        pnl_col = f"pnl_bps_{horizon}s"
        if pnl_col in joined.columns:
            cols.append(pnl_col)
    out = joined.select([col for col in cols if col in joined.columns]).head(20)
    return out.with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="us").dt.strftime("%Y-%m-%dT%H:%M:%S%.6fZ").alias(
            "timestamp_utc"
        )
    )


def daily_stability(markouts: pl.DataFrame, horizons_seconds: tuple[int, ...]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon in horizons_seconds:
        pnl = f"pnl_bps_{horizon}s"
        if pnl not in markouts.columns:
            continue
        df = markouts.filter(pl.col(pnl).is_not_null())
        if df.height == 0:
            continue
        daily = (
            df.group_by(["date", "split", "symbol"])
            .agg(
                pl.len().alias("sample_rows"),
                pl.col("clipped_notional").sum().alias("sample_clipped_turnover"),
                ((pl.col(pnl) * pl.col("clipped_notional")).sum() / pl.col("clipped_notional").sum()).alias(
                    "weighted_pnl_bps"
                ),
            )
            .with_columns(pl.lit(horizon).alias("horizon_seconds"))
        )
        rows.extend(daily.to_dicts())
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def nonlinear_flow_response(markouts: pl.DataFrame, horizons_seconds: tuple[int, ...]) -> pl.DataFrame:
    if "signed_liq_notional_30s" not in markouts.columns:
        return pl.DataFrame()
    rows: list[dict[str, object]] = []
    for horizon in horizons_seconds:
        pnl = f"pnl_bps_{horizon}s"
        if pnl not in markouts.columns:
            continue
        valid = markouts.filter(pl.col(pnl).is_not_null() & pl.col("signed_liq_notional_30s").is_not_null())
        if valid.height < 20:
            continue
        for symbol, part in valid.group_by("symbol", maintain_order=True):
            symbol_value = symbol[0] if isinstance(symbol, tuple) else symbol
            if part.height < 20:
                continue
            zero = part.filter(pl.col("signed_liq_notional_30s").abs() <= 1e-12).with_columns(
                pl.lit("zero").alias("signed_liq_quantile")
            )
            nonzero = part.filter(pl.col("signed_liq_notional_30s").abs() > 1e-12)
            buckets = [zero] if zero.height else []
            if 0 < nonzero.height < 20:
                buckets.append(nonzero.with_columns(pl.lit("nonzero_sparse").alias("signed_liq_quantile")))
            elif nonzero.height >= 20:
                buckets.append(
                    nonzero.sort("signed_liq_notional_30s")
                    .with_row_index("_rank")
                    .with_columns(
                        (
                            ((pl.col("_rank") * 10) / pl.len())
                            .floor()
                            .clip(upper_bound=9)
                            .cast(pl.Int64)
                            + 1
                        )
                        .cast(pl.Utf8)
                        .alias("signed_liq_quantile")
                    )
                    .drop("_rank")
                )
            bucketed = pl.concat(buckets, how="diagonal_relaxed") if buckets else pl.DataFrame()
            if bucketed.height == 0:
                continue
            rows.extend(
                bucketed.group_by("signed_liq_quantile")
                .agg(
                    pl.lit(symbol_value).alias("symbol"),
                    pl.len().alias("rows"),
                    pl.col("signed_liq_notional_30s").mean().alias(
                        "mean_signed_liq_notional_30s"
                    ),
                    pl.col(pnl).mean().alias("mean_pnl_bps"),
                    (
                        (pl.col(pnl) * pl.col("clipped_notional")).sum()
                        / pl.col("clipped_notional").sum()
                    ).alias("weighted_pnl_bps"),
                )
                .with_columns(pl.lit(horizon).alias("horizon_seconds"))
                .to_dicts()
            )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: object) -> float:
    try:
        result = float(value)
    except Exception:
        return math.nan
    return result
