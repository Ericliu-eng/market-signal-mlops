from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import JSON, DateTime, Float, String, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Mapped, Session, mapped_column

from market_signal_mlops.monitoring.service import MonitoringReport
from market_signal_mlops.storage.predictions import Base


class MonitoringRow(Base):
    __tablename__ = "monitoring_reports"

    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        primary_key=True,
    )
    latest_event_ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    freshness_hours: Mapped[float] = mapped_column(Float, nullable=False)
    missing_rate: Mapped[float] = mapped_column(Float, nullable=False)
    feature_psi: Mapped[dict[str, float]] = mapped_column(
        JSON,
        nullable=False,
    )
    maximum_feature_psi: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    recommendation: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    reasons: Mapped[list[str]] = mapped_column(JSON, nullable=False)


def _as_utc_timestamp(value: datetime) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        return timestamp.tz_localize(timezone.utc)

    return timestamp.tz_convert(timezone.utc)


class MonitoringRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    def save_report(self, report: MonitoringReport) -> None:
        with Session(self.engine) as session:
            session.merge(
                MonitoringRow(
                    observed_at=report.observed_at.to_pydatetime(),
                    latest_event_ts=report.latest_event_ts.to_pydatetime(),
                    freshness_hours=report.freshness_hours,
                    missing_rate=report.missing_rate,
                    feature_psi=report.feature_psi,
                    maximum_feature_psi=report.maximum_feature_psi,
                    status=report.status,
                    recommendation=report.recommendation,
                    reasons=list(report.reasons),
                )
            )
            session.commit()

    def get_latest_report(self) -> MonitoringReport | None:
        statement = (
            select(MonitoringRow).order_by(MonitoringRow.observed_at.desc()).limit(1)
        )

        with Session(self.engine) as session:
            row = session.scalar(statement)

            if row is None:
                return None

            return MonitoringReport(
                observed_at=_as_utc_timestamp(row.observed_at),
                latest_event_ts=_as_utc_timestamp(row.latest_event_ts),
                freshness_hours=row.freshness_hours,
                missing_rate=row.missing_rate,
                feature_psi={
                    str(name): float(value) for name, value in row.feature_psi.items()
                },
                maximum_feature_psi=row.maximum_feature_psi,
                status=row.status,
                recommendation=row.recommendation,
                reasons=tuple(row.reasons),
            )
