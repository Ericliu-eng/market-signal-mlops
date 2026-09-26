from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from market_signal_mlops.inference.batch import build_prediction_records


class FakeClassifier:
    feature_names_in_ = np.array(["feature_a", "feature_b"])
    classes_ = np.array([0, 1])

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        assert list(features.columns) == ["feature_a", "feature_b"]
        return np.array([1, 0])

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        assert list(features.columns) == ["feature_a", "feature_b"]
        return np.array(
            [
                [0.20, 0.80],
                [0.70, 0.30],
            ]
        )


def test_build_prediction_records_uses_latest_row_per_symbol() -> None:
    feature_snapshot = pd.DataFrame(
        [
            {
                "event_ts": pd.Timestamp("2026-09-23", tz="UTC"),
                "symbol": "AAPL",
                "snapshot_id": "snapshot-1",
                "feature_set_version": "pit_features_v1",
                "feature_a": 1.0,
                "feature_b": 2.0,
            },
            {
                "event_ts": pd.Timestamp("2026-09-24", tz="UTC"),
                "symbol": "AAPL",
                "snapshot_id": "snapshot-1",
                "feature_set_version": "pit_features_v1",
                "feature_a": 3.0,
                "feature_b": 4.0,
            },
            {
                "event_ts": pd.Timestamp("2026-09-23", tz="UTC"),
                "symbol": "MSFT",
                "snapshot_id": "snapshot-1",
                "feature_set_version": "pit_features_v1",
                "feature_a": 5.0,
                "feature_b": 6.0,
            },
            {
                "event_ts": pd.Timestamp("2026-09-24", tz="UTC"),
                "symbol": "MSFT",
                "snapshot_id": "snapshot-1",
                "feature_set_version": "pit_features_v1",
                "feature_a": 7.0,
                "feature_b": 8.0,
            },
        ]
    )
    generated_at = datetime(2026, 9, 24, tzinfo=timezone.utc)

    records = build_prediction_records(
        model=FakeClassifier(),
        feature_snapshot=feature_snapshot,
        model_name="market-signal-classifier",
        model_version="1",
        generated_at=generated_at,
    )

    assert len(records) == 2
    assert [record.symbol for record in records] == ["AAPL", "MSFT"]
    assert all(
        record.as_of_date.isoformat() == "2026-09-24"
        for record in records
    )
    assert records[0].prediction == 1
    assert records[0].probability == pytest.approx(0.80)
    assert records[1].prediction == 0
    assert records[1].probability == pytest.approx(0.30)
    assert all(record.model_version == "1" for record in records)
    assert all(record.snapshot_id == "snapshot-1" for record in records)