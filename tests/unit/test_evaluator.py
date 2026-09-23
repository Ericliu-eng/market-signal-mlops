from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from market_signal_mlops.evaluation.evaluator import (
    EvaluationConfig,
    TimeSeriesEvaluator,
)
from market_signal_mlops.features.feature_builder import build_feature_snapshot
from market_signal_mlops.features.labels import build_next_day_direction_labels


@pytest.fixture
def evaluation_config() -> EvaluationConfig:
    return EvaluationConfig(
        target_column="target_next_day_direction",
        min_train_size=3,
        validation_size=2,
        n_splits=3,
    )


@pytest.fixture
def modeling_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    event_ts = pd.date_range("2026-01-01", periods=30, freq="D")
    close = [100.0]
    for index in range(1, len(event_ts)):
        close.append(close[-1] * (1.01 if index % 2 else 0.99))

    bars = pd.DataFrame(
        {
            "event_ts": event_ts,
            "symbol": "AAPL",
            "open": close,
            "high": [value * 1.01 for value in close],
            "low": [value * 0.99 for value in close],
            "close": close,
            "volume": [1_000_000 + (index * 1_000) for index in range(len(event_ts))],
            "source": "unit-test",
            "snapshot_id": "snapshot-2026-01-30",
            "ingested_at": event_ts + pd.Timedelta(hours=1),
        }
    )
    return build_feature_snapshot(bars), build_next_day_direction_labels(bars)


def test_evaluator_returns_one_result_per_model_per_fold(
    evaluation_config: EvaluationConfig,
    modeling_inputs: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    result = TimeSeriesEvaluator(evaluation_config).evaluate(*modeling_inputs)

    expected_rows = 3 * evaluation_config.n_splits
    assert len(result.fold_metrics) == expected_rows
    assert set(result.fold_metrics["model_name"]) == {
        "naive_prior",
        "logistic_regression",
        "hist_gradient_boosting",
    }
    pairs = result.predictions[["fold_number", "model_name"]].drop_duplicates()
    assert len(pairs) == expected_rows


def test_each_fold_is_strictly_chronological(
    evaluation_config: EvaluationConfig,
    modeling_inputs: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    result = TimeSeriesEvaluator(evaluation_config).evaluate(*modeling_inputs)

    assert (
        result.fold_boundaries["train_end"]
        < result.fold_boundaries["validation_start"]
    ).all()


def test_evaluation_is_deterministic(
    evaluation_config: EvaluationConfig,
    modeling_inputs: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    first = TimeSeriesEvaluator(evaluation_config).evaluate(*modeling_inputs)
    second = TimeSeriesEvaluator(evaluation_config).evaluate(*modeling_inputs)

    assert_frame_equal(first.fold_metrics, second.fold_metrics)
    assert_frame_equal(first.fold_boundaries, second.fold_boundaries)
    assert_frame_equal(first.predictions, second.predictions)


def test_evaluator_rejects_missing_target(
    evaluation_config: EvaluationConfig,
    modeling_inputs: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    feature_snapshot, labels = modeling_inputs
    labels_without_target = labels.drop(columns=[evaluation_config.target_column])

    with pytest.raises(ValueError, match="target"):
        TimeSeriesEvaluator(evaluation_config).evaluate(
            feature_snapshot,
            labels_without_target,
        )

def test_fit_candidate_model_returns_fitted_model(
    evaluation_config: EvaluationConfig,
    modeling_inputs: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    evaluator = TimeSeriesEvaluator(evaluation_config)

    model, training_features = evaluator.fit_candidate_model(
        "hist_gradient_boosting",
        *modeling_inputs,
    )

    predictions = model.predict(training_features)

    assert len(predictions) == len(training_features)
    assert set(predictions).issubset({0, 1})