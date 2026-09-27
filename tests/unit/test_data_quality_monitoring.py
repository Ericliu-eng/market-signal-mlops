import pandas as pd
import pytest

from market_signal_mlops.monitoring.data_quality import (
    calculate_freshness_hours,
    calculate_missing_rate,
)


def test_calculate_freshness_hours() -> None:
    latest_event_ts = pd.Timestamp("2026-06-18T18:00:00Z")
    observed_at = pd.Timestamp("2026-06-19T00:00:00Z")

    result = calculate_freshness_hours(
        latest_event_ts=latest_event_ts,
        observed_at=observed_at,
    )

    assert result == 6.0


def test_future_event_timestamp_is_rejected() -> None:
    latest_event_ts = pd.Timestamp("2026-06-19T01:00:00Z")
    observed_at = pd.Timestamp("2026-06-19T00:00:00Z")

    with pytest.raises(ValueError, match="future"):
        calculate_freshness_hours(
            latest_event_ts=latest_event_ts,
            observed_at=observed_at,
        )


def test_calculate_missing_rate_for_selected_features() -> None:
    frame = pd.DataFrame(
        {
            "return_1d": [0.01, None],
            "volume_change_1d": [None, 0.20],
            "symbol": ["AAPL", "MSFT"],
        }
    )

    result = calculate_missing_rate(
        frame,
        feature_columns=["return_1d", "volume_change_1d"],
    )

    assert result == 0.5


def test_missing_feature_column_is_rejected() -> None:
    frame = pd.DataFrame({"return_1d": [0.01]})

    with pytest.raises(ValueError, match="missing feature columns"):
        calculate_missing_rate(
            frame,
            feature_columns=["return_1d", "volume_change_1d"],
        )