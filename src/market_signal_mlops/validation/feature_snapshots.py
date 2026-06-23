from __future__ import annotations

import numpy as np
import pandas as pd

from market_signal_mlops.contracts.schemas import (
    FEATURE_SNAPSHOT_DATETIME_COLUMNS,
    FEATURE_SNAPSHOT_PRIMARY_KEY_COLUMNS,
    FEATURE_SNAPSHOT_REQUIRED_COLUMNS,
    FEATURE_SNAPSHOT_STRING_COLUMNS,
)


def validate_feature_snapshot(df: pd.DataFrame) -> None:
    missing_columns = [
        column
        for column in FEATURE_SNAPSHOT_REQUIRED_COLUMNS
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"missing required columns: {missing_columns}")

    null_required_columns = [
        column
        for column in FEATURE_SNAPSHOT_REQUIRED_COLUMNS
        if df[column].isnull().any()
    ]
    if null_required_columns:
        raise ValueError(f"null required values: {null_required_columns}")

    duplicated_primary_keys = df.duplicated(
        subset=FEATURE_SNAPSHOT_PRIMARY_KEY_COLUMNS
    ).any()
    if duplicated_primary_keys:
        raise ValueError("duplicate primary keys")

    for column in FEATURE_SNAPSHOT_DATETIME_COLUMNS:
        if not pd.api.types.is_datetime64_any_dtype(df[column]):
            raise ValueError(f"{column} must be datetime")

    for column in FEATURE_SNAPSHOT_STRING_COLUMNS:
        if not df[column].map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"{column} must contain string values")

        if df[column].str.strip().eq("").any():
            raise ValueError(f"{column} cannot contain blank values")

    sorted_df = df.sort_values(["symbol", "event_ts"]).reset_index(drop=True)
    current_df = df.reset_index(drop=True)

    if not current_df[["symbol", "event_ts"]].equals(sorted_df[["symbol", "event_ts"]]):
        raise ValueError("rows must be ordered by symbol and event_ts")

    metadata_columns = set(FEATURE_SNAPSHOT_REQUIRED_COLUMNS)
    feature_columns = [
        column for column in df.columns if column not in metadata_columns
    ]

    if not feature_columns:
        raise ValueError("at least one feature column is required")

    for column in feature_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"{column} must be numeric")

        if not np.isfinite(df[column]).all():
            raise ValueError(f"{column} must contain finite values")
