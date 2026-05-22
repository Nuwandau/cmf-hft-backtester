from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from cmf_backtester.liquidation.config import LiquidationEdaConfig
from cmf_backtester.liquidation.schema import EXPECTED_DTYPES, SourceSpec, source_specs


def ensure_output_dirs(config: LiquidationEdaConfig) -> None:
    config.output_root.mkdir(parents=True, exist_ok=True)
    config.tables_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    config.processed_root.mkdir(parents=True, exist_ok=True)


def source_path(config: LiquidationEdaConfig, spec: SourceSpec) -> Path:
    return config.raw_root / spec.relative_path


def scan_source(config: LiquidationEdaConfig, spec: SourceSpec) -> pl.LazyFrame:
    return pl.scan_parquet(source_path(config, spec))


def timestamp_to_utc_string(value_us: int | None) -> str | None:
    if value_us is None:
        return None
    return datetime.fromtimestamp(value_us / 1_000_000, tz=timezone.utc).isoformat()


def date_expr(column: str = "timestamp") -> pl.Expr:
    return pl.from_epoch(pl.col(column), time_unit="us").dt.strftime("%Y-%m-%d")


def hour_expr(column: str = "timestamp") -> pl.Expr:
    return pl.from_epoch(pl.col(column), time_unit="us").dt.strftime("%H")


def collect_frame(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect(streaming=True)


def source_file_table(config: LiquidationEdaConfig) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in source_specs(config.symbols):
        path = source_path(config, spec)
        rows.append(
            {
                "source": spec.source,
                "venue": spec.venue,
                "data_type": spec.data_type,
                "symbol": spec.symbol,
                "ticker": spec.ticker,
                "path": path.as_posix(),
                "size_mb": path.stat().st_size / 1_000_000 if path.exists() else 0.0,
                "exists": path.exists(),
            }
        )
    return pl.DataFrame(rows)


def schema_audit(config: LiquidationEdaConfig) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in source_specs(config.symbols):
        path = source_path(config, spec)
        if not path.exists():
            rows.append(
                {
                    "source": spec.source,
                    "symbol": spec.symbol,
                    "ok": False,
                    "issue": "missing_file",
                    "columns": "",
                    "dtypes": "",
                    "expected_columns": ",".join(spec.columns),
                }
            )
            continue
        schema = scan_source(config, spec).collect_schema()
        columns = list(schema.names())
        dtypes = [schema[col] for col in columns]
        missing = [col for col in spec.columns if col not in columns]
        unexpected = [col for col in columns if col not in spec.columns]
        dtype_issues = [
            f"{col}:{schema[col]}!={EXPECTED_DTYPES[col]}"
            for col in spec.columns
            if col in schema and col in EXPECTED_DTYPES and schema[col] != EXPECTED_DTYPES[col]
        ]
        issue_parts = []
        if missing:
            issue_parts.append(f"missing={missing}")
        if unexpected:
            issue_parts.append(f"unexpected={unexpected}")
        if dtype_issues:
            issue_parts.append(f"dtype={dtype_issues}")
        rows.append(
            {
                "source": spec.source,
                "venue": spec.venue,
                "data_type": spec.data_type,
                "symbol": spec.symbol,
                "ok": not issue_parts,
                "issue": "; ".join(issue_parts),
                "columns": ",".join(columns),
                "dtypes": ",".join(str(dtype) for dtype in dtypes),
                "expected_columns": ",".join(spec.columns),
            }
        )
    return pl.DataFrame(rows)


def source_quality_tables(config: LiquidationEdaConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    coverage_rows: list[dict[str, Any]] = []
    range_rows: list[dict[str, Any]] = []
    for spec in source_specs(config.symbols):
        lf = scan_source(config, spec)
        select_exprs = [
            pl.len().alias("rows"),
            pl.col("timestamp").min().alias("min_timestamp"),
            pl.col("timestamp").max().alias("max_timestamp"),
            pl.col("timestamp").n_unique().alias("unique_timestamps"),
            (pl.len() - pl.col("timestamp").n_unique()).alias("duplicate_timestamp_rows"),
        ]
        for col in spec.columns:
            select_exprs.append(pl.col(col).null_count().alias(f"{col}_nulls"))
        if "price" in spec.columns:
            select_exprs.extend(
                [
                    pl.col("price").min().alias("min_price"),
                    pl.col("price").max().alias("max_price"),
                    pl.col("amount").min().alias("min_amount"),
                    pl.col("amount").max().alias("max_amount"),
                ]
            )
        else:
            select_exprs.extend(
                [
                    pl.col("bid_price").min().alias("min_bid_price"),
                    pl.col("ask_price").max().alias("max_ask_price"),
                    pl.col("bid_amount").min().alias("min_bid_amount"),
                    pl.col("ask_amount").min().alias("min_ask_amount"),
                    pl.col("bid_amount").max().alias("max_bid_amount"),
                    pl.col("ask_amount").max().alias("max_ask_amount"),
                ]
            )
        row = collect_frame(lf.select(select_exprs)).row(0, named=True)
        row.update(
            {
                "source": spec.source,
                "venue": spec.venue,
                "data_type": spec.data_type,
                "symbol": spec.symbol,
                "min_datetime_utc": timestamp_to_utc_string(row["min_timestamp"]),
                "max_datetime_utc": timestamp_to_utc_string(row["max_timestamp"]),
            }
        )
        coverage_rows.append(row)
        if "side" in spec.columns:
            sides = collect_frame(lf.group_by("side").agg(pl.len().alias("rows"))).with_columns(
                pl.lit(spec.source).alias("source"),
                pl.lit(spec.symbol).alias("symbol"),
            )
            range_rows.extend(sides.to_dicts())
    return pl.DataFrame(coverage_rows), pl.DataFrame(range_rows) if range_rows else pl.DataFrame()


def daily_event_counts(config: LiquidationEdaConfig) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for spec in source_specs(config.symbols):
        df = collect_frame(
            scan_source(config, spec)
            .select(
                date_expr().alias("date"),
                pl.lit(spec.source).alias("source"),
                pl.lit(spec.venue).alias("venue"),
                pl.lit(spec.data_type).alias("data_type"),
                pl.lit(spec.symbol).alias("symbol"),
            )
            .group_by(["source", "venue", "data_type", "symbol", "date"])
            .agg(pl.len().alias("rows"))
            .sort(["source", "symbol", "date"])
        )
        frames.append(df)
    return pl.concat(frames, how="diagonal_relaxed")


def hourly_event_counts(config: LiquidationEdaConfig) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for spec in source_specs(config.symbols):
        df = collect_frame(
            scan_source(config, spec)
            .select(
                hour_expr().alias("hour_utc"),
                pl.lit(spec.source).alias("source"),
                pl.lit(spec.venue).alias("venue"),
                pl.lit(spec.data_type).alias("data_type"),
                pl.lit(spec.symbol).alias("symbol"),
            )
            .group_by(["source", "venue", "data_type", "symbol", "hour_utc"])
            .agg(pl.len().alias("rows"))
            .sort(["source", "symbol", "hour_utc"])
        )
        frames.append(df)
    return pl.concat(frames, how="diagonal_relaxed")


def deterministic_sample(
    config: LiquidationEdaConfig,
    spec: SourceSpec,
    max_rows: int,
    columns: list[str] | None = None,
) -> pl.DataFrame:
    lf = scan_source(config, spec)
    if columns is not None:
        lf = lf.select([col for col in columns if col in spec.columns])
    if config.profile == "quick":
        return collect_frame(lf.with_row_index("original_row_id").limit(max_rows))
    count = int(collect_frame(lf.select(pl.len().alias("rows")))["rows"][0])
    stride = max(1, count // max(max_rows, 1))
    return collect_frame(
        lf.with_row_index("original_row_id")
        .filter((pl.col("original_row_id") % stride) == 0)
        .limit(max_rows)
    )
