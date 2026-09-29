import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

from market_signal_mlops.features.feature_builder import (
    build_feature_snapshot,
)
from market_signal_mlops.monitoring.runner import (
    evaluate_and_store_monitoring,
)
from market_signal_mlops.storage.monitoring import MonitoringRepository


DEFAULT_DATABASE_URL = (
    "postgresql+psycopg://market_signal:market_signal"
    "@localhost:5433/market_signal"
)
DEFAULT_MARKET_BARS_PATH = Path("data/fixtures/market_bars.csv")
DEFAULT_CURRENT_WINDOW_SIZE = 20

METADATA_COLUMNS = {
    "event_ts",
    "symbol",
    "snapshot_id",
    "feature_set_version",
    "generated_at",
}


def select_feature_columns(frame: pd.DataFrame) -> list[str]:
    feature_columns = sorted(
        column
        for column in frame.columns
        if column not in METADATA_COLUMNS
        and pd.api.types.is_numeric_dtype(frame[column])
    )

    if not feature_columns:
        raise ValueError("no numeric feature columns available")

    return feature_columns


def split_monitoring_windows(
    feature_snapshot: pd.DataFrame,
    *,
    current_window_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if current_window_size < 1:
        raise ValueError("current_window_size must be positive")

    if "event_ts" not in feature_snapshot.columns:
        raise ValueError("feature snapshot must contain event_ts")

    event_timestamps = (
        pd.to_datetime(feature_snapshot["event_ts"], utc=True)
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if len(event_timestamps) <= current_window_size:
        raise ValueError(
            "current window must leave data for the reference window"
        )

    cutoff = event_timestamps.iloc[-current_window_size]
    timestamps = pd.to_datetime(feature_snapshot["event_ts"], utc=True)

    reference_frame = feature_snapshot.loc[timestamps < cutoff].copy()
    current_frame = feature_snapshot.loc[timestamps >= cutoff].copy()

    return reference_frame, current_frame


def _observed_at_from_environment() -> pd.Timestamp:
    configured = os.getenv("MONITORING_OBSERVED_AT")

    if configured:
        timestamp = pd.Timestamp(configured)
    else:
        timestamp = pd.Timestamp.now(tz="UTC")

    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")

    return timestamp.tz_convert("UTC")


def main() -> None:
    database_url = os.getenv(
        "PREDICTION_DATABASE_URL",
        DEFAULT_DATABASE_URL,
    )
    market_bars_path = Path(
        os.getenv(
            "MARKET_BARS_PATH",
            str(DEFAULT_MARKET_BARS_PATH),
        )
    )
    current_window_size = int(
        os.getenv(
            "MONITORING_CURRENT_WINDOW_SIZE",
            str(DEFAULT_CURRENT_WINDOW_SIZE),
        )
    )
    observed_at = _observed_at_from_environment()

    market_bars = pd.read_csv(
        market_bars_path,
        parse_dates=["event_ts", "ingested_at"],
    )
    feature_snapshot = build_feature_snapshot(
        market_bars,
        generated_at=observed_at,
    )
    reference_frame, current_frame = split_monitoring_windows(
        feature_snapshot,
        current_window_size=current_window_size,
    )
    feature_columns = select_feature_columns(feature_snapshot)

    engine = create_engine(database_url)
    try:
        repository = MonitoringRepository(engine)
        repository.create_schema()
        report = evaluate_and_store_monitoring(
            reference_frame=reference_frame,
            current_frame=current_frame,
            feature_columns=feature_columns,
            observed_at=observed_at,
            repository=repository,
        )
    finally:
        engine.dispose()

    print(
        "Monitoring completed: "
        f"status={report.status}, "
        f"recommendation={report.recommendation}, "
        f"freshness_hours={report.freshness_hours:.2f}, "
        f"maximum_feature_psi={report.maximum_feature_psi:.4f}"
    )


if __name__ == "__main__":
    main()