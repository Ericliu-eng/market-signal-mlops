import pandas as pd
import pytest

from market_signal_mlops.monitoring.run import (
    select_feature_columns,
    split_monitoring_windows,
)


def _snapshot() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_ts": pd.to_datetime(
                [
                    "2026-06-01",
                    "2026-06-02",
                    "2026-06-03",
                    "2026-06-04",
                    "2026-06-05",
                ],
                utc=True,
            ),
            "symbol": ["AAPL"] * 5,
            "snapshot_id": ["market_bars"] * 5,
            "feature_set_version": ["pit_features_v1"] * 5,
            "generated_at": pd.to_datetime(
                ["2026-06-06T00:00:00Z"] * 5,
                utc=True,
            ),
            "return_1d": [0.01, 0.02, 0.03, 0.04, 0.05],
            "volume_change_1d": [0.10, 0.20, 0.30, 0.40, 0.50],
        }
    )


def test_split_monitoring_windows_keeps_time_order() -> None:
    reference, current = split_monitoring_windows(
        _snapshot(),
        current_window_size=2,
    )

    assert reference["event_ts"].max() < current["event_ts"].min()
    assert current["event_ts"].nunique() == 2
    assert len(reference) == 3
    assert len(current) == 2


def test_window_size_must_leave_reference_data() -> None:
    with pytest.raises(ValueError, match="reference"):
        split_monitoring_windows(
            _snapshot(),
            current_window_size=5,
        )


def test_select_feature_columns_excludes_metadata() -> None:
    result = select_feature_columns(_snapshot())

    assert result == ["return_1d", "volume_change_1d"]