from __future__ import annotations

LOB_LEVELS = 25
TIMESTAMP_COL = "local_timestamp"
INDEX_COL = ""


def ask_price_col(level: int) -> str:
    return f"asks[{level}].price"


def ask_amount_col(level: int) -> str:
    return f"asks[{level}].amount"


def bid_price_col(level: int) -> str:
    return f"bids[{level}].price"


def bid_amount_col(level: int) -> str:
    return f"bids[{level}].amount"


def required_l1_columns() -> list[str]:
    return [
        TIMESTAMP_COL,
        ask_price_col(0),
        ask_amount_col(0),
        bid_price_col(0),
        bid_amount_col(0),
    ]


def required_lob_columns(levels: int = LOB_LEVELS) -> list[str]:
    cols = [TIMESTAMP_COL]
    for level in range(levels):
        cols.extend(
            [
                ask_price_col(level),
                ask_amount_col(level),
                bid_price_col(level),
                bid_amount_col(level),
            ]
        )
    return cols


def validate_columns(columns: set[str] | list[str], required: list[str]) -> None:
    present = set(columns)
    missing = [col for col in required if col not in present]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

