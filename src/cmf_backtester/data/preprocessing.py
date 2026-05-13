from __future__ import annotations

from pathlib import Path

import polars as pl

from cmf_backtester.data.schema import (
    ask_amount_col,
    ask_price_col,
    bid_amount_col,
    bid_price_col,
    required_l1_columns,
    validate_columns,
)
from cmf_backtester.data.splitting import split_expr


def _scan_table(path: str | Path) -> pl.LazyFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pl.scan_parquet(path)
    if path.suffix == ".csv":
        return pl.scan_csv(path, infer_schema_length=1000)
    raise ValueError(f"Unsupported data format: {path.suffix}")


def _with_event_id(lf: pl.LazyFrame) -> pl.LazyFrame:
    columns = lf.collect_schema().names()
    if "event_id" in columns:
        return lf.with_columns(pl.col("event_id").cast(pl.Int64))
    if "" in columns:
        return lf.rename({"": "event_id"}).with_columns(pl.col("event_id").cast(pl.Int64))
    return lf.with_row_index("event_id")


def estimate_tick_size_from_lob(raw_lob_path: str | Path, sample_rows: int | None = None) -> float:
    """Estimate tick size from top-of-book prices using Polars."""
    lf = _scan_table(raw_lob_path).select(
        pl.col(ask_price_col(0)).cast(pl.Float64).alias("ask"),
        pl.col(bid_price_col(0)).cast(pl.Float64).alias("bid"),
    )
    if sample_rows is not None:
        lf = lf.head(sample_rows)
    prices = (
        lf.select(pl.concat_list("ask", "bid").alias("prices"))
        .explode("prices")
        .select((pl.col("prices") * 10_000_000).round(0).cast(pl.Int64).alias("p"))
        .unique()
        .sort("p")
        .collect()
    )
    vals = prices["p"].to_list()
    diffs = [b - a for a, b in zip(vals, vals[1:]) if b > a]
    if not diffs:
        raise ValueError("Could not estimate tick size from LOB prices")
    return min(diffs) / 10_000_000


def preprocess_lob_l1(
    raw_lob_path: str | Path,
    output_path: str | Path,
    tick_size: float,
    train_dates: list[str],
    validation_dates: list[str],
    test_dates: list[str],
) -> None:
    """Convert raw LOB data into a compact event-ordered L1 Parquet dataset."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lf = _with_event_id(_scan_table(raw_lob_path))
    validate_columns(lf.collect_schema().names(), required_l1_columns())

    tick = pl.lit(float(tick_size))
    processed = (
        lf.select(
            pl.col("event_id").cast(pl.Int64),
            pl.col("local_timestamp").cast(pl.Int64),
            pl.col(ask_price_col(0)).cast(pl.Float64).alias("best_ask"),
            pl.col(bid_price_col(0)).cast(pl.Float64).alias("best_bid"),
            pl.col(ask_amount_col(0)).cast(pl.Float64).alias("ask_size"),
            pl.col(bid_amount_col(0)).cast(pl.Float64).alias("bid_size"),
        )
        .with_columns(
            (pl.col("best_ask") / tick).round(0).cast(pl.Int64).alias("best_ask_ticks"),
            (pl.col("best_bid") / tick).round(0).cast(pl.Int64).alias("best_bid_ticks"),
            pl.from_epoch(pl.col("local_timestamp"), time_unit="us").dt.date().alias("date"),
        )
        .with_columns(
            (pl.col("best_bid_ticks") + pl.col("best_ask_ticks")).alias("mid_half_ticks"),
            ((pl.col("best_bid_ticks") + pl.col("best_ask_ticks")) / 2.0).alias("mid_ticks"),
            (pl.col("best_ask_ticks") - pl.col("best_bid_ticks")).alias("spread_ticks"),
            (
                pl.col("bid_size") / (pl.col("bid_size") + pl.col("ask_size"))
            ).alias("imbalance"),
        )
        .with_columns(split_expr(train_dates, validation_dates, test_dates))
        .sort("event_id")
        .select(
            "event_id",
            "local_timestamp",
            "date",
            "split",
            "best_bid",
            "best_ask",
            "best_bid_ticks",
            "best_ask_ticks",
            "bid_size",
            "ask_size",
            "mid_ticks",
            "mid_half_ticks",
            "spread_ticks",
            "imbalance",
        )
    )
    processed.sink_parquet(output_path)


def create_sample_data(
    processed_lob_path: str | Path,
    sample_output_path: str | Path,
    n_rows: int = 10_000,
) -> None:
    sample_output_path = Path(sample_output_path)
    sample_output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.scan_parquet(processed_lob_path).head(n_rows).collect()
    df.write_parquet(sample_output_path)


def audit_processed_lob(processed_lob_path: str | Path) -> dict[str, object]:
    df = pl.scan_parquet(processed_lob_path)
    summary = df.select(
        pl.len().alias("rows"),
        (pl.col("local_timestamp").diff() < 0).sum().alias("timestamp_order_violations"),
        (pl.len() - pl.col("local_timestamp").n_unique()).alias("duplicate_timestamps"),
        pl.col("local_timestamp").min().alias("first_timestamp"),
        pl.col("local_timestamp").max().alias("last_timestamp"),
        pl.col("spread_ticks").min().alias("min_spread_ticks"),
        pl.col("spread_ticks").median().alias("median_spread_ticks"),
        pl.col("spread_ticks").quantile(0.9).alias("p90_spread_ticks"),
        pl.col("spread_ticks").quantile(0.99).alias("p99_spread_ticks"),
        pl.col("spread_ticks").max().alias("max_spread_ticks"),
        pl.col("imbalance").mean().alias("mean_imbalance"),
    ).collect()
    row = summary.row(0, named=True)
    split_counts = (
        df.group_by("split")
        .agg(pl.len().alias("rows"))
        .sort("split")
        .collect()
        .to_dicts()
    )
    row["split_counts"] = split_counts
    return row


def audit_processed_lob_by_date(processed_lob_path: str | Path) -> pl.DataFrame:
    df = pl.scan_parquet(processed_lob_path)
    return (
        df.group_by("date", "split")
        .agg(
            pl.len().alias("rows"),
            pl.col("spread_ticks").median().alias("median_spread_ticks"),
            pl.col("spread_ticks").quantile(0.9).alias("p90_spread_ticks"),
            pl.col("spread_ticks").quantile(0.99).alias("p99_spread_ticks"),
            (pl.col("spread_ticks") == 1).mean().alias("fraction_one_tick_spread"),
            (pl.col("spread_ticks") > 10).mean().alias("fraction_spread_gt_10_ticks"),
            pl.col("imbalance").mean().alias("mean_imbalance"),
        )
        .sort("date")
        .collect()
    )
