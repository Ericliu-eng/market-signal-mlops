from typing import Protocol

import numpy as np
import pandas as pd

from market_signal_mlops.monitoring.runner import (
    evaluate_and_store_monitoring,
)
from market_signal_mlops.monitoring.service import MonitoringReport


class ReportWriter(Protocol):
    def save_report(self, report: MonitoringReport) -> None: ...


class FakeMonitoringRepository:
    def __init__(self) -> None:
        self.saved_report: MonitoringReport | None = None

    def save_report(self, report: MonitoringReport) -> None:
        self.saved_report = report


def _frame(values: np.ndarray, *, event_ts: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_ts": pd.to_datetime([event_ts] * len(values), utc=True),
            "return_1d": values,
            "volume_change_1d": values * 2,
        }
    )


def test_evaluate_and_store_monitoring() -> None:
    values = np.arange(1, 101, dtype=float)
    reference = _frame(
        values,
        event_ts="2026-06-18T00:00:00Z",
    )
    current = _frame(
        values.copy(),
        event_ts="2026-06-18T18:00:00Z",
    )
    repository = FakeMonitoringRepository()

    report = evaluate_and_store_monitoring(
        reference_frame=reference,
        current_frame=current,
        feature_columns=["return_1d", "volume_change_1d"],
        observed_at=pd.Timestamp("2026-06-19T00:00:00Z"),
        repository=repository,
    )

    assert report.status == "healthy"
    assert repository.saved_report is report