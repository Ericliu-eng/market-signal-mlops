from dataclasses import replace
from datetime import date, datetime, timezone

import pytest
from sqlalchemy import create_engine

from market_signal_mlops.inference.schemas import PredictionRecord
from market_signal_mlops.storage.predictions import PredictionRepository


def make_prediction() -> PredictionRecord:
    return PredictionRecord(
        as_of_date=date(2026, 9, 24),
        symbol="AAPL",
        prediction=1,
        probability=0.78,
        model_name="market-signal-classifier",
        model_version="1",
        feature_version="pit_features_v1",
        snapshot_id="market_bars",
        generated_at=datetime(2026, 9, 24, tzinfo=timezone.utc),
    )


def test_upsert_is_idempotent() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    repository = PredictionRepository(engine)
    repository.create_schema()

    prediction = make_prediction()
    repository.upsert_predictions([prediction])
    repository.upsert_predictions(
        [replace(prediction, probability=0.91)]
    )

    stored = repository.list_predictions()

    assert len(stored) == 1
    assert stored[0].probability == pytest.approx(0.91)