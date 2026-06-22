from __future__ import annotations

import pandas as pd
import numpy as np

from market_signal_mlops.contracts.schemas import (
    MARKET_BAR_REQUIRED_COLUMNS,
    PRICE_COLUMNS,
    PRIMARY_KEY_COLUMNS,
    DATETIME_COLUMNS,
    STRING_COLUMNS,
)


def validate_market_bars(df: pd.DataFrame) -> None:
    """
    Validate the input market bars data contract.

    This function raises ValueError when the input data violates the contract.
    If no error is raised, the data is considered valid.
    """

    missing_columns = [
        col for col in MARKET_BAR_REQUIRED_COLUMNS if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df[MARKET_BAR_REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError("Required columns contain null values.")

    for column in DATETIME_COLUMNS:
        if not pd.api.types.is_datetime64_any_dtype(df[column]):
            raise ValueError(f"{column} must be datetime")

    duplicated_rows = df.duplicated(subset=PRIMARY_KEY_COLUMNS)

    if duplicated_rows.any():
        raise ValueError("Duplicate primary key found: event_ts + symbol.")

    # string_column_requires_string_values
    for column in STRING_COLUMNS:
        if not pd.api.types.is_string_dtype(df[column]):
            raise ValueError(f"{column} must be string")
        if (df[column].astype(str).str.strip() == "").any():
            raise ValueError(f"{column} must be not null")

    for column in PRICE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"{column} must be number")

        if not np.isfinite(df[column]).all():
            raise ValueError(f"{column} must contain finite values")

        if (df[column] <= 0).any():
            raise ValueError(f"Price column must be positive: {column}")
    for symbol, group in df.groupby("symbol", sort=False):
        if not group["event_ts"].is_monotonic_increasing:
            raise ValueError("rows must be ordered by symbol and event_ts")

    if not pd.api.types.is_integer_dtype(df["volume"]):
        raise ValueError("volume must be integer")
    if (df["volume"] < 0).any():
        raise ValueError("Volume must be non-negative.")

    expected = df.sort_values(["symbol", "event_ts"]).reset_index(drop=True)
    actual = df.reset_index(drop=True)

    if not actual.equals(expected):
        raise ValueError("rows must be ordered by symbol and event_ts")

    invalid_ohlc = (
        (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["high"] < df["low"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
        | (df["low"] > df["high"])
    )

    if invalid_ohlc.any():
        raise ValueError("invalid OHLC relationship")
