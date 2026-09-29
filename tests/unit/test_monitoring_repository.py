from datetime import timezone

import pandas as pd
from sqlalchemy import create_engine

from market_signal_mlops.monitoring.service import MonitoringReport
from market_signal_mlops.storage.monitoring import MonitoringRepository


def _report(
    *,
    observed_at: str,
    status: str,
) -> MonitoringReport:
    return MonitoringReport(
        observed_at=pd.Timestamp(observed_at),
        latest_event_ts=pd.Timestamp("2026-06-18T18:00:00Z"),
        freshness_hours=6.0,
        missing_rate=0.01,
        feature_psi={
            "return_1d": 0.05,
            "volume_change_1d": 0.08,
        },
        maximum_feature_psi=0.08,
        status=status,
        recommendation="no_action",
        reasons=(),
    )


def test_save_and_load_latest_monitoring_report() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    repository = MonitoringRepository(engine)
    repository.create_schema()

    repository.save_report(
        _report(
            observed_at="2026-06-19T00:00:00Z",
            status="healthy",
        )
    )
    repository.save_report(
        _report(
            observed_at="2026-06-20T00:00:00Z",
            status="warning",
        )
    )

    result = repository.get_latest_report()

    assert result is not None
    assert result.observed_at.tzinfo == timezone.utc
    assert result.status == "warning"
    assert result.feature_psi == {
        "return_1d": 0.05,
        "volume_change_1d": 0.08,
    }

    engine.dispose()


def test_empty_repository_has_no_latest_report() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    repository = MonitoringRepository(engine)
    repository.create_schema()

    assert repository.get_latest_report() is None

    engine.dispose()