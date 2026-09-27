from collections.abc import Sequence

import pandas as pd


def _as_utc_timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")

    return timestamp.tz_convert("UTC")


def calculate_freshness_hours(
    *,
    latest_event_ts: pd.Timestamp | str,
    observed_at: pd.Timestamp | str,
) -> float:
    latest = _as_utc_timestamp(latest_event_ts)
    observed = _as_utc_timestamp(observed_at)

    if latest > observed:
        raise ValueError("latest event timestamp cannot be in the future")

    return (observed - latest).total_seconds() / 3600


def calculate_missing_rate(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
) -> float:
    if not feature_columns:
        raise ValueError("at least one feature column is required")

    missing_columns = [
        column for column in feature_columns if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(f"missing feature columns: {missing_columns}")

    if frame.empty:
        raise ValueError("cannot calculate missing rate for an empty frame")

    return float(frame[list(feature_columns)].isna().to_numpy().mean())