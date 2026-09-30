import pandas as pd
import pytest

from market_signal_mlops.monitoring.performance import (
    calculate_model_performance,
)


def test_calculate_model_performance() -> None:
    frame = pd.DataFrame(
        {
            "as_of_date": pd.to_datetime(
                ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04"]
            ),
            "actual": [0, 1, 1, 0],
            "prediction": [0, 1, 1, 0],
            "probability": [0.1, 0.9, 0.8, 0.2],
        }
    )

    result = calculate_model_performance(frame)

    assert result.sample_count == 4
    assert result.balanced_accuracy == 1.0
    assert result.f1 == 1.0
    assert result.brier_score == pytest.approx(0.025)
    assert result.roc_auc == 1.0


def test_unavailable_labels_are_excluded() -> None:
    frame = pd.DataFrame(
        {
            "as_of_date": pd.to_datetime(["2026-06-01", "2026-06-02", "2026-06-03"]),
            "actual": [1, None, 0],
            "prediction": [1, 0, 0],
            "probability": [0.8, 0.4, 0.2],
        }
    )

    result = calculate_model_performance(frame)

    assert result.sample_count == 2
    assert result.balanced_accuracy == 1.0


def test_rolling_window_uses_latest_labeled_rows() -> None:
    frame = pd.DataFrame(
        {
            "as_of_date": pd.to_datetime(
                ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04"]
            ),
            "actual": [0, 1, 0, 1],
            "prediction": [1, 0, 0, 1],
            "probability": [0.8, 0.2, 0.1, 0.9],
        }
    )

    result = calculate_model_performance(frame, window_size=2)

    assert result.sample_count == 2
    assert result.balanced_accuracy == 1.0
    assert result.f1 == 1.0


def test_performance_requires_at_least_one_label() -> None:
    frame = pd.DataFrame(
        {
            "as_of_date": pd.to_datetime(["2026-06-01"]),
            "actual": [None],
            "prediction": [1],
            "probability": [0.8],
        }
    )

    with pytest.raises(ValueError, match="labeled"):
        calculate_model_performance(frame)
