from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ModelPerformance:
    sample_count: int
    balanced_accuracy: float
    f1: float
    brier_score: float
    roc_auc: float | None


def calculate_model_performance(
    frame: pd.DataFrame,
    *,
    window_size: int | None = None,
) -> ModelPerformance:
    required_columns = {
        "as_of_date",
        "actual",
        "prediction",
        "probability",
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"missing performance columns: {missing_columns}")

    if window_size is not None and window_size < 1:
        raise ValueError("window_size must be positive")

    labeled = frame.loc[frame["actual"].notna()].copy()
    if labeled.empty:
        raise ValueError("at least one labeled prediction is required")

    labeled["as_of_date"] = pd.to_datetime(
        labeled["as_of_date"],
        utc=True,
    )
    labeled = labeled.sort_values("as_of_date")

    if window_size is not None:
        labeled = labeled.tail(window_size)

    actual = labeled["actual"].astype(int)
    prediction = labeled["prediction"].astype(int)
    probability = labeled["probability"].astype(float)

    if not actual.isin([0, 1]).all():
        raise ValueError("actual values must be 0 or 1")
    if not prediction.isin([0, 1]).all():
        raise ValueError("prediction values must be 0 or 1")
    if not probability.between(0.0, 1.0).all():
        raise ValueError("probability values must be between 0 and 1")
    if not np.isfinite(probability).all():
        raise ValueError("probability values must be finite")

    roc_auc: float | None = None
    if actual.nunique() == 2:
        roc_auc = float(roc_auc_score(actual, probability))

    return ModelPerformance(
        sample_count=len(labeled),
        balanced_accuracy=float(balanced_accuracy_score(actual, prediction)),
        f1=float(f1_score(actual, prediction, zero_division=0)),
        brier_score=float(brier_score_loss(actual, probability)),
        roc_auc=roc_auc,
    )
