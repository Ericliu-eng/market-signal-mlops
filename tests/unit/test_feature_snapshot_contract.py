from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_signal_mlops.validation.feature_snapshots import validate_feature_snapshot


def _valid_feature_snapshot() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_ts": pd.to_datetime(
                [
                    "2024-01-01 09:30:00",
                    "2024-01-02 09:30:00",
                ]
            ),
            "symbol": ["AAPL", "AAPL"],
            "snapshot_id": ["snapshot-001", "snapshot-001"],
            "feature_set_version": ["v1", "v1"],
            "generated_at": pd.to_datetime(
                [
                    "2024-01-02 10:00:00",
                    "2024-01-02 10:00:00",
                ]
            ),
            "return_5d": [0.01, 0.02],
        }
    )


def test_valid_feature_snapshot_passes_contract() -> None:
    df = _valid_feature_snapshot()

    validate_feature_snapshot(df)


def test_missing_required_column_fails() -> None:
    df = _valid_feature_snapshot().drop(columns=["feature_set_version"])

    with pytest.raises(ValueError, match="missing required columns"):
        validate_feature_snapshot(df)


def test_null_required_value_fails() -> None:
    df = _valid_feature_snapshot()
    df.loc[0, "snapshot_id"] = None

    with pytest.raises(ValueError, match="null required values"):
        validate_feature_snapshot(df)


def test_duplicate_primary_key_fails() -> None:
    df = _valid_feature_snapshot()
    df.loc[1, "event_ts"] = df.loc[0, "event_ts"]

    with pytest.raises(ValueError, match="duplicate primary keys"):
        validate_feature_snapshot(df)


@pytest.mark.parametrize("column", ["event_ts", "generated_at"])
def test_datetime_column_requires_datetime_dtype(column: str) -> None:
    df = _valid_feature_snapshot()
    df[column] = df[column].astype(str)

    with pytest.raises(ValueError, match="datetime"):
        validate_feature_snapshot(df)


@pytest.mark.parametrize("column", ["symbol", "snapshot_id", "feature_set_version"])
def test_string_column_requires_string_values(column: str) -> None:
    df = _valid_feature_snapshot()
    df[column] = df[column].astype(object)
    df.loc[0, column] = 123

    with pytest.raises(ValueError, match="string"):
        validate_feature_snapshot(df)


@pytest.mark.parametrize("column", ["symbol", "snapshot_id", "feature_set_version"])
def test_string_column_rejects_blank_values(column: str) -> None:
    df = _valid_feature_snapshot()
    df.loc[0, column] = "   "

    with pytest.raises(ValueError, match="blank"):
        validate_feature_snapshot(df)


def test_rows_must_be_ordered_by_symbol_and_event_ts() -> None:
    df = _valid_feature_snapshot().iloc[[1, 0]].reset_index(drop=True)

    with pytest.raises(ValueError, match="ordered"):
        validate_feature_snapshot(df)


def test_requires_at_least_one_feature_column() -> None:
    df = _valid_feature_snapshot().drop(columns=["return_5d"])

    with pytest.raises(ValueError, match="feature column"):
        validate_feature_snapshot(df)


def test_feature_column_requires_numeric_dtype() -> None:
    df = _valid_feature_snapshot()
    df["return_5d"] = ["bad", "data"]

    with pytest.raises(ValueError, match="numeric"):
        validate_feature_snapshot(df)


@pytest.mark.parametrize("bad_value", [np.inf, -np.inf])
def test_feature_column_rejects_non_finite_values(bad_value: float) -> None:
    df = _valid_feature_snapshot()
    df.loc[0, "return_5d"] = bad_value

    with pytest.raises(ValueError, match="finite"):
        validate_feature_snapshot(df)
