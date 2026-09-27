import numpy as np
import pandas as pd
import pytest

from market_signal_mlops.monitoring.service import build_monitoring_report


def _feature_frame(
    values: np.ndarray,
    *,
    event_ts: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_ts": pd.to_datetime([event_ts] * len(values), utc=True),
            "return_1d": values,
            "volume_change_1d": values * 2,
        }
    )


def test_build_healthy_monitoring_report() -> None:
    values = np.arange(1, 101, dtype=float)
    reference = _feature_frame(
        values,
        event_ts="2026-06-18T00:00:00Z",
    )
    current = _feature_frame(
        values.copy(),
        event_ts="2026-06-18T18:00:00Z",
    )

    report = build_monitoring_report(
        reference_frame=reference,
        current_frame=current,
        feature_columns=["return_1d", "volume_change_1d"],
        observed_at=pd.Timestamp("2026-06-19T00:00:00Z"),
    )

    assert report.freshness_hours == 6.0
    assert report.missing_rate == 0.0
    assert report.maximum_feature_psi == pytest.approx(0.0)
    assert report.status == "healthy"
    assert report.recommendation == "no_action"
    assert report.reasons == ()


def test_build_report_recommends_retraining_for_drift() -> None:
    reference_values = np.arange(1, 101, dtype=float)
    current_values = np.arange(101, 201, dtype=float)

    reference = _feature_frame(
        reference_values,
        event_ts="2026-06-18T00:00:00Z",
    )
    current = _feature_frame(
        current_values,
        event_ts="2026-06-18T18:00:00Z",
    )

    report = build_monitoring_report(
        reference_frame=reference,
        current_frame=current,
        feature_columns=["return_1d", "volume_change_1d"],
        observed_at=pd.Timestamp("2026-06-19T00:00:00Z"),
    )

    assert report.maximum_feature_psi > 0.25
    assert report.status == "critical"
    assert report.recommendation == "recommend_retrain"
    assert "feature drift exceeds critical threshold" in report.reasons