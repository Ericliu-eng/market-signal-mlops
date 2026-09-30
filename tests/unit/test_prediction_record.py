from datetime import date, datetime, timezone

import pytest

from market_signal_mlops.inference.schemas import PredictionRecord


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


def test_valid_prediction_record_is_created() -> None:
    prediction = make_prediction()

    assert prediction.symbol == "AAPL"
    assert prediction.prediction == 1
    assert prediction.probability == pytest.approx(0.78)


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_probability_must_be_between_zero_and_one(probability: float) -> None:
    with pytest.raises(ValueError, match="probability"):
        PredictionRecord(
            as_of_date=date(2026, 9, 24),
            symbol="AAPL",
            prediction=1,
            probability=probability,
            model_name="market-signal-classifier",
            model_version="1",
            feature_version="pit_features_v1",
            snapshot_id="market_bars",
            generated_at=datetime(2026, 9, 24, tzinfo=timezone.utc),
        )


def test_prediction_must_be_binary() -> None:
    with pytest.raises(ValueError, match="prediction"):
        PredictionRecord(
            as_of_date=date(2026, 9, 24),
            symbol="AAPL",
            prediction=2,
            probability=0.78,
            model_name="market-signal-classifier",
            model_version="1",
            feature_version="pit_features_v1",
            snapshot_id="market_bars",
            generated_at=datetime(2026, 9, 24, tzinfo=timezone.utc),
        )
