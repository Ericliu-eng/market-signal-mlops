import os
from dataclasses import replace
from datetime import date, datetime, timezone

import pytest
from sqlalchemy import create_engine, text

from market_signal_mlops.inference.schemas import PredictionRecord
from market_signal_mlops.storage.predictions import PredictionRepository


TEST_DATE = date(2099, 1, 1)
TEST_SYMBOL = "WEEK6_TEST"


def delete_test_rows(engine) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                "DELETE FROM predictions "
                "WHERE as_of_date = :as_of_date AND symbol = :symbol"
            ),
            {
                "as_of_date": TEST_DATE,
                "symbol": TEST_SYMBOL,
            },
        )


def test_postgres_upsert_is_idempotent() -> None:
    database_url = os.getenv("PREDICTION_DATABASE_URL")
    if database_url is None:
        pytest.skip("PREDICTION_DATABASE_URL is not configured")

    engine = create_engine(database_url)
    repository = PredictionRepository(engine)
    repository.create_schema()
    delete_test_rows(engine)

    prediction = PredictionRecord(
        as_of_date=TEST_DATE,
        symbol=TEST_SYMBOL,
        prediction=1,
        probability=0.78,
        model_name="market-signal-classifier",
        model_version="1",
        feature_version="pit_features_v1",
        snapshot_id="integration-test",
        generated_at=datetime.now(timezone.utc),
    )

    try:
        repository.upsert_predictions([prediction])
        repository.upsert_predictions([replace(prediction, probability=0.91)])

        stored = repository.list_predictions(
            as_of_date=TEST_DATE,
            symbol=TEST_SYMBOL,
        )

        assert len(stored) == 1
        assert stored[0].probability == pytest.approx(0.91)
    finally:
        delete_test_rows(engine)
        engine.dispose()
