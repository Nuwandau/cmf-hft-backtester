from __future__ import annotations

import polars as pl

from cmf_backtester.liquidation.features import add_trade_features, classify_trade_locations


def join_trades_to_bbo(trades: pl.DataFrame, bbo: pl.DataFrame, tolerance_us: int) -> pl.DataFrame:
    left = add_trade_features(trades).sort("timestamp")
    right = bbo.sort("timestamp")
    joined = left.join_asof(
        right.select(
            [
                pl.col("timestamp").alias("bbo_timestamp"),
                "bid_price",
                "bid_amount",
                "ask_price",
                "ask_amount",
                "mid",
                "spread_bps",
                "queue_imbalance",
            ]
        ),
        left_on="timestamp",
        right_on="bbo_timestamp",
        strategy="backward",
        tolerance=tolerance_us,
    ).with_columns((pl.col("timestamp") - pl.col("bbo_timestamp")).alias("bbo_age_us"))
    return classify_trade_locations(joined)


def compute_markouts(
    trades_with_bbo: pl.DataFrame,
    bbo: pl.DataFrame,
    horizons_seconds: tuple[int, ...],
    maker_rebate_bps: float,
    tolerance_us: int,
) -> pl.DataFrame:
    result = trades_with_bbo
    future_bbo = bbo.select(
        [
            pl.col("timestamp").alias("future_bbo_timestamp"),
            pl.col("mid").alias("future_mid"),
        ]
    ).sort("future_bbo_timestamp")
    for horizon in horizons_seconds:
        target_col = f"target_timestamp_{horizon}s"
        future_mid_col = f"future_mid_{horizon}s"
        future_ts_col = f"future_bbo_timestamp_{horizon}s"
        age_col = f"future_bbo_age_us_{horizon}s"
        pnl_col = f"pnl_bps_{horizon}s"
        result = result.with_columns((pl.col("timestamp") + horizon * 1_000_000).alias(target_col))
        joined = result.sort(target_col).join_asof(
            future_bbo,
            left_on=target_col,
            right_on="future_bbo_timestamp",
            strategy="backward",
            tolerance=tolerance_us,
        )
        result = (
            joined.rename(
                {
                    "future_mid": future_mid_col,
                    "future_bbo_timestamp": future_ts_col,
                }
            )
            .with_columns((pl.col(target_col) - pl.col(future_ts_col)).alias(age_col))
            .with_columns(
                (
                    -pl.col("maker_direction")
                    * (pl.col(future_mid_col) - pl.col("price"))
                    / pl.col("price")
                    * 10_000
                    + maker_rebate_bps
                ).alias(pnl_col)
            )
            .sort("timestamp")
        )
    return result


def summarize_markouts(df: pl.DataFrame, horizons_seconds: tuple[int, ...]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    groups = ["symbol", "side"]
    if "split" in df.columns:
        groups = ["split", *groups]
    for group_values, part in df.group_by(groups, maintain_order=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        base = dict(zip(groups, group_values, strict=True))
        for horizon in horizons_seconds:
            pnl = f"pnl_bps_{horizon}s"
            valid = part.filter(pl.col(pnl).is_not_null())
            if valid.height == 0:
                continue
            weights = valid["clipped_notional"]
            weighted = float((valid[pnl] * weights).sum() / weights.sum())
            row = {
                **base,
                "horizon_seconds": horizon,
                "rows": valid.height,
                "weighted_pnl_bps": weighted,
                "mean_pnl_bps": float(valid[pnl].mean()),
                "median_pnl_bps": float(valid[pnl].median()),
                "p10_pnl_bps": float(valid[pnl].quantile(0.10)),
                "p90_pnl_bps": float(valid[pnl].quantile(0.90)),
                "clipped_turnover": float(weights.sum()),
            }
            rows.append(row)
    return pl.DataFrame(rows) if rows else pl.DataFrame()
