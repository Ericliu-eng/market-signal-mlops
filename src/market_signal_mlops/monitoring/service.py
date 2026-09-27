from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from market_signal_mlops.monitoring.data_quality import (
    calculate_freshness_hours,
    calculate_missing_rate,
)
from market_signal_mlops.monitoring.drift import calculate_psi
from market_signal_mlops.monitoring.policy import (
    MonitoringEvidence,
    MonitoringPolicy,
)


@dataclass(frozen=True)
class MonitoringReport:
    observed_at: pd.Timestamp
    latest_event_ts: pd.Timestamp
    freshness_hours: float
    missing_rate: float
    feature_psi: dict[str, float]
    maximum_feature_psi: float
    status: str
    recommendation: str
    reasons: tuple[str, ...]


def build_monitoring_report(
    *,
    reference_frame: pd.DataFrame,
    current_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    observed_at: pd.Timestamp | str,
    policy: MonitoringPolicy | None = None,
) -> MonitoringReport:
    if current_frame.empty:
        raise ValueError("current frame cannot be empty")

    if "event_ts" not in current_frame.columns:
        raise ValueError("current frame must contain event_ts")

    missing_reference_columns = [
        column for column in feature_columns if column not in reference_frame.columns
    ]
    if missing_reference_columns:
        raise ValueError(
            f"reference frame is missing feature columns: {missing_reference_columns}"
        )

    observed_timestamp = pd.Timestamp(observed_at)
    latest_event_timestamp = pd.to_datetime(
        current_frame["event_ts"],
        utc=True,
    ).max()

    if pd.isna(latest_event_timestamp):
        raise ValueError("current frame must contain a valid event timestamp")

    freshness_hours = calculate_freshness_hours(
        latest_event_ts=latest_event_timestamp,
        observed_at=observed_timestamp,
    )
    missing_rate = calculate_missing_rate(
        current_frame,
        feature_columns=feature_columns,
    )

    feature_psi: dict[str, float] = {}

    for column in feature_columns:
        reference_values = reference_frame[column].dropna().to_numpy()
        current_values = current_frame[column].dropna().to_numpy()

        if reference_values.size == 0 or current_values.size == 0:
            feature_psi[column] = 0.0
            continue

        feature_psi[column] = calculate_psi(
            reference_values,
            current_values,
        )

    maximum_feature_psi = max(feature_psi.values(), default=0.0)

    decision = (policy or MonitoringPolicy()).evaluate(
        MonitoringEvidence(
            freshness_hours=freshness_hours,
            missing_rate=missing_rate,
            feature_psi=maximum_feature_psi,
        )
    )

    return MonitoringReport(
        observed_at=observed_timestamp,
        latest_event_ts=latest_event_timestamp,
        freshness_hours=freshness_hours,
        missing_rate=missing_rate,
        feature_psi=feature_psi,
        maximum_feature_psi=maximum_feature_psi,
        status=decision.status,
        recommendation=decision.recommendation,
        reasons=decision.reasons,
    )