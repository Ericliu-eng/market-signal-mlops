import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sqlalchemy import create_engine

from market_signal_mlops.features.feature_builder import (
    build_feature_snapshot,
)
from market_signal_mlops.inference.batch import build_prediction_records
from market_signal_mlops.storage.predictions import PredictionRepository


REGISTERED_MODEL_NAME = "market-signal-classifier"
MODEL_ALIAS = "champion"
DEFAULT_TRACKING_URI = "http://localhost:5000"
DEFAULT_DATABASE_URL = (
    "postgresql+psycopg://market_signal:market_signal@localhost:5433/market_signal"
)
DEFAULT_MARKET_BARS_PATH = Path("data/fixtures/market_bars.csv")


def main() -> None:
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        DEFAULT_TRACKING_URI,
    )
    database_url = os.getenv(
        "PREDICTION_DATABASE_URL",
        DEFAULT_DATABASE_URL,
    )
    market_bars_path = Path(
        os.getenv("MARKET_BARS_PATH", str(DEFAULT_MARKET_BARS_PATH))
    )

    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    champion = client.get_model_version_by_alias(
        REGISTERED_MODEL_NAME,
        MODEL_ALIAS,
    )

    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.sklearn.load_model(model_uri)

    market_bars = pd.read_csv(
        market_bars_path,
        parse_dates=["event_ts", "ingested_at"],
    )
    feature_snapshot = build_feature_snapshot(market_bars)

    records = build_prediction_records(
        model=model,
        feature_snapshot=feature_snapshot,
        model_name=REGISTERED_MODEL_NAME,
        model_version=str(champion.version),
    )

    engine = create_engine(database_url)
    try:
        repository = PredictionRepository(engine)
        repository.create_schema()
        repository.upsert_predictions(records)
    finally:
        engine.dispose()

    print(
        f"Stored {len(records)} predictions from "
        f"{REGISTERED_MODEL_NAME} version {champion.version}."
    )


if __name__ == "__main__":
    main()
