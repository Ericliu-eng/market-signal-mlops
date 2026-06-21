from __future__ import annotations

import pandas as pd

from market_signal_mlops.contracts.schemas import (
    MARKET_BAR_REQUIRED_COLUMNS,
    PRICE_COLUMNS,
    PRIMARY_KEY_COLUMNS,
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

    duplicated_rows = df.duplicated(subset=PRIMARY_KEY_COLUMNS)

    if duplicated_rows.any():
        raise ValueError("Duplicate primary key found: event_ts + symbol.")

    for col in PRICE_COLUMNS:
        if (df[col] <= 0).any():
            raise ValueError(f"Price column must be positive: {col}")

    if (df["volume"] < 0).any():
        raise ValueError("Volume must be non-negative.")

    df_sorted = df.sort_values(["symbol", "event_ts"])

    for symbol, group in df_sorted.groupby("symbol"):
        if not group["event_ts"].is_monotonic_increasing:
            raise ValueError(f"event_ts must be increasing for symbol: {symbol}")