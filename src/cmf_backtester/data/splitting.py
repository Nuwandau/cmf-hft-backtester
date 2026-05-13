from __future__ import annotations

import numpy as np
import polars as pl


def split_expr(
    train_dates: list[str],
    validation_dates: list[str],
    test_dates: list[str],
    date_col: str = "date",
) -> pl.Expr:
    """Build a Polars expression that assigns chronological split labels."""
    date_str = pl.col(date_col).cast(pl.Utf8)
    return (
        pl.when(date_str.is_in(train_dates))
        .then(pl.lit("train"))
        .when(date_str.is_in(validation_dates))
        .then(pl.lit("validation"))
        .when(date_str.is_in(test_dates))
        .then(pl.lit("test"))
        .otherwise(pl.lit("unused"))
        .alias("split")
    )


def split_mask(split_array: np.ndarray, split_name: str) -> np.ndarray:
    return split_array == split_name


def date_mask(date_array: np.ndarray, dates: list[str]) -> np.ndarray:
    return np.isin(date_array, np.asarray(dates, dtype=str))

