from collections.abc import Iterable
from datetime import date, timezone

from sqlalchemy import Date, DateTime, Float, Integer, String, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from market_signal_mlops.inference.schemas import PredictionRecord


class Base(DeclarativeBase):
    pass


class PredictionRow(Base):
    __tablename__ = "predictions"

    as_of_date: Mapped[date] = mapped_column(Date, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), primary_key=True)
    model_name: Mapped[str] = mapped_column(String(128), primary_key=True)
    model_version: Mapped[str] = mapped_column(String(32), primary_key=True)
    feature_version: Mapped[str] = mapped_column(String(64), primary_key=True)

    prediction: Mapped[int] = mapped_column(Integer, nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    generated_at: Mapped[date] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )


class PredictionRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    def upsert_predictions(
        self,
        predictions: Iterable[PredictionRecord],
    ) -> None:
        with Session(self.engine) as session:
            for prediction in predictions:
                session.merge(
                    PredictionRow(
                        as_of_date=prediction.as_of_date,
                        symbol=prediction.symbol,
                        prediction=prediction.prediction,
                        probability=prediction.probability,
                        model_name=prediction.model_name,
                        model_version=prediction.model_version,
                        feature_version=prediction.feature_version,
                        snapshot_id=prediction.snapshot_id,
                        generated_at=prediction.generated_at,
                    )
                )
            session.commit()

    def list_predictions(
        self,
        *,
        as_of_date: date | None = None,
        symbol: str | None = None,
    ) -> list[PredictionRecord]:
        statement = select(PredictionRow)

        if as_of_date is not None:
            statement = statement.where(PredictionRow.as_of_date == as_of_date)
        if symbol is not None:
            statement = statement.where(PredictionRow.symbol == symbol)

        statement = statement.order_by(
            PredictionRow.as_of_date,
            PredictionRow.symbol,
            PredictionRow.model_version,
        )

        with Session(self.engine) as session:
            rows = session.scalars(statement).all()

            records = []
            for row in rows:
                generated_at = row.generated_at
                if generated_at.tzinfo is None:
                    generated_at = generated_at.replace(tzinfo=timezone.utc)

                records.append(
                    PredictionRecord(
                        as_of_date=row.as_of_date,
                        symbol=row.symbol,
                        prediction=row.prediction,
                        probability=row.probability,
                        model_name=row.model_name,
                        model_version=row.model_version,
                        feature_version=row.feature_version,
                        snapshot_id=row.snapshot_id,
                        generated_at=generated_at,
                    )
                )

            return records
