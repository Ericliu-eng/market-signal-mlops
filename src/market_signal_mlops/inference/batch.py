from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from market_signal_mlops.inference.schemas import PredictionRecord


REQUIRED_METADATA_COLUMNS = {
    "event_ts",
    "symbol",
    "snapshot_id",
    "feature_set_version",
}


def build_prediction_records(
    *,
    model: Any,
    feature_snapshot: pd.DataFrame,
    model_name: str,
    model_version: str,
    generated_at: datetime | None = None,
) -> list[PredictionRecord]:
    missing_columns = REQUIRED_METADATA_COLUMNS - set(feature_snapshot.columns)
    if missing_columns:
        raise ValueError(
            f"feature snapshot missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if feature_snapshot.empty:
        raise ValueError("feature snapshot must not be empty")

    if generated_at is None:
        generated_at = datetime.now(timezone.utc)

    latest_features = (
        feature_snapshot.copy()
        .sort_values(["symbol", "event_ts"], kind="stable")
        .groupby("symbol", as_index=False, sort=False)
        .tail(1)
        .sort_values("symbol", kind="stable")
        .reset_index(drop=True)
    )

    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        raise ValueError("model does not expose feature_names_in_")

    feature_columns = [str(column) for column in feature_names]
    missing_features = [
        column
        for column in feature_columns
        if column not in latest_features.columns
    ]
    if missing_features:
        raise ValueError(
            f"feature snapshot missing model inputs: {missing_features}"
        )

    model_inputs = latest_features[feature_columns]
    predictions = np.asarray(model.predict(model_inputs))
    probabilities = np.asarray(model.predict_proba(model_inputs))

    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("model does not contain positive class 1")
    positive_class_index = classes.index(1)

    records = []
    for row_number, row in latest_features.iterrows():
        event_ts = pd.Timestamp(row["event_ts"])

        records.append(
            PredictionRecord(
                as_of_date=event_ts.date(),
                symbol=str(row["symbol"]),
                prediction=int(predictions[row_number]),
                probability=float(
                    probabilities[row_number, positive_class_index]
                ),
                model_name=model_name,
                model_version=model_version,
                feature_version=str(row["feature_set_version"]),
                snapshot_id=str(row["snapshot_id"]),
                generated_at=generated_at,
            )
        )

    return records