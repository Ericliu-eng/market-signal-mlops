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
    """Purpose: 最小合法 FeatureSnapshot 应该通过 contract。"""
    df = _valid_feature_snapshot()

    validate_feature_snapshot(df)


def test_missing_required_column_fails() -> None:
    """Purpose: 缺少 metadata 必填列时，contract 必须拒绝。"""
    df = _valid_feature_snapshot().drop(columns=["feature_set_version"])

    with pytest.raises(ValueError, match="missing required columns"):
        validate_feature_snapshot(df)


def test_null_required_value_fails() -> None:
    """Purpose: metadata 必填列不能有 null，否则快照不可追踪。"""
    df = _valid_feature_snapshot()
    df.loc[0, "snapshot_id"] = None

    with pytest.raises(ValueError, match="null required values"):
        validate_feature_snapshot(df)


def test_duplicate_primary_key_fails() -> None:
    """Purpose: 同一个 event_ts + symbol + snapshot_id + feature_set_version 不能重复。"""
    df = _valid_feature_snapshot()
    df.loc[1, "event_ts"] = df.loc[0, "event_ts"]

    with pytest.raises(ValueError, match="duplicate primary keys"):
        validate_feature_snapshot(df)


@pytest.mark.parametrize("column", ["event_ts", "generated_at"])
def test_datetime_column_requires_datetime_dtype(column: str) -> None:
    """Purpose: 时间字段必须是真正 datetime dtype，不能只是字符串。"""
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
    """Purpose: 字符串 metadata 字段不能是空白字符串。"""
    df = _valid_feature_snapshot()
    df.loc[0, column] = "   "

    with pytest.raises(ValueError, match="blank"):
        validate_feature_snapshot(df)


def test_rows_must_be_ordered_by_symbol_and_event_ts() -> None:
    """Purpose: 快照行要稳定排序，避免训练/回测输入顺序不确定。"""
    df = _valid_feature_snapshot().iloc[[1, 0]].reset_index(drop=True)

    with pytest.raises(ValueError, match="ordered"):
        validate_feature_snapshot(df)


def test_requires_at_least_one_feature_column() -> None:
    """Purpose: FeatureSnapshot 不能只有 metadata，必须至少有一个动态 feature。"""
    df = _valid_feature_snapshot().drop(columns=["return_5d"])

    with pytest.raises(ValueError, match="feature column"):
        validate_feature_snapshot(df)


def test_feature_column_requires_numeric_dtype() -> None:
    """Purpose: 动态 feature 列应该是数值型，方便后续模型训练。"""
    df = _valid_feature_snapshot()
    df["return_5d"] = ["bad", "data"]

    with pytest.raises(ValueError, match="numeric"):
        validate_feature_snapshot(df)


@pytest.mark.parametrize("bad_value", [np.inf, -np.inf])
def test_feature_column_rejects_non_finite_values(bad_value: float) -> None:
    """Purpose: 动态 feature 不能有 inf/-inf，否则训练会炸。"""
    df = _valid_feature_snapshot()
    df.loc[0, "return_5d"] = bad_value

    with pytest.raises(ValueError, match="finite"):
        validate_feature_snapshot(df)
