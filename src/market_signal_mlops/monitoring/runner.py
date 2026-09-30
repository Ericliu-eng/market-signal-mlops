from collections.abc import Sequence
from typing import Protocol

import pandas as pd

from market_signal_mlops.monitoring.service import (
    MonitoringReport,
    build_monitoring_report,
)


class MonitoringReportWriter(Protocol):
    def save_report(self, report: MonitoringReport) -> None: ...


def evaluate_and_store_monitoring(
    *,
    reference_frame: pd.DataFrame,
    current_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    observed_at: pd.Timestamp | str,
    repository: MonitoringReportWriter,
) -> MonitoringReport:
    report = build_monitoring_report(
        reference_frame=reference_frame,
        current_frame=current_frame,
        feature_columns=feature_columns,
        observed_at=observed_at,
    )

    repository.save_report(report)

    return report
