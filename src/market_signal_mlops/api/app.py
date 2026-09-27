import os
from datetime import date, datetime
from typing import Protocol

from fastapi import FastAPI
from mlflow import MlflowClient
from pydantic import BaseModel, ConfigDict
from sqlalchemy import create_engine

from market_signal_mlops.inference.schemas import PredictionRecord
from market_signal_mlops.storage.predictions import PredictionRepository


REGISTERED_MODEL_NAME = "market-signal-classifier"
MODEL_ALIAS = "champion"
DEFAULT_TRACKING_URI = "http://localhost:5000"
DEFAULT_DATABASE_URL = (
    "postgresql+psycopg://market_signal:market_signal@localhost:5433/market_signal"
)


class PredictionReader(Protocol):
    def list_predictions(
        self,
        *,
        as_of_date: date | None = None,
        symbol: str | None = None,
    ) -> list[PredictionRecord]: ...


class ModelVersionInfo(Protocol):
    version: str
    status: str
    run_id: str


class ModelRegistryReader(Protocol):
    def get_model_version_by_alias(
        self,
        name: str,
        alias: str,
    ) -> ModelVersionInfo: ...


class PredictionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    as_of_date: date
    symbol: str
    prediction: int
    probability: float
    model_name: str
    model_version: str
    feature_version: str
    snapshot_id: str
    generated_at: datetime


class ModelResponse(BaseModel):
    model_name: str
    alias: str
    version: str
    status: str
    run_id: str


class MonitoringSummaryResponse(BaseModel):
    total_predictions: int
    latest_as_of_date: date | None
    symbols: list[str]
    model_versions: list[str]
    feature_versions: list[str]


def create_app(
    prediction_repository: PredictionReader | None = None,
    model_registry: ModelRegistryReader | None = None,
) -> FastAPI:
    application = FastAPI(
        title="Market Signal API",
        version="0.1.0",
    )

    if prediction_repository is None:
        database_url = os.getenv(
            "PREDICTION_DATABASE_URL",
            DEFAULT_DATABASE_URL,
        )
        engine = create_engine(database_url)
        prediction_repository = PredictionRepository(engine)
        application.state.prediction_engine = engine

    if model_registry is None:
        tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI",
            DEFAULT_TRACKING_URI,
        )
        model_registry = MlflowClient(tracking_uri=tracking_uri)

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get(
        "/predictions",
        response_model=list[PredictionResponse],
    )
    def predictions(
        as_of_date: date | None = None,
        symbol: str | None = None,
    ) -> list[PredictionRecord]:
        return prediction_repository.list_predictions(
            as_of_date=as_of_date,
            symbol=symbol,
        )

    @application.get(
        "/model",
        response_model=ModelResponse,
    )
    def model() -> ModelResponse:
        champion = model_registry.get_model_version_by_alias(
            REGISTERED_MODEL_NAME,
            MODEL_ALIAS,
        )
        return ModelResponse(
            model_name=REGISTERED_MODEL_NAME,
            alias=MODEL_ALIAS,
            version=str(champion.version),
            status=str(champion.status),
            run_id=str(champion.run_id),
        )

    @application.get(
        "/monitoring-summary",
        response_model=MonitoringSummaryResponse,
    )
    def monitoring_summary() -> MonitoringSummaryResponse:
        records = prediction_repository.list_predictions()

        latest_as_of_date = max(
            (record.as_of_date for record in records),
            default=None,
        )

        return MonitoringSummaryResponse(
            total_predictions=len(records),
            latest_as_of_date=latest_as_of_date,
            symbols=sorted({record.symbol for record in records}),
            model_versions=sorted({record.model_version for record in records}),
            feature_versions=sorted({record.feature_version for record in records}),
        )

    return application


app = create_app()
