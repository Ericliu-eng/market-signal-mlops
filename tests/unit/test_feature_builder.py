# tests/unit/test_feature_builder.py

from __future__ import annotations

import pandas as pd
import pytest

from market_signal_mlops.features.feature_builder import build_feature_snapshot
from market_signal_mlops.validation.feature_snapshots import validate_feature_snapshot

GENERATED_AT = pd.Timestamp("2026-01-06T00:00:00Z")

def make_market_bars(row_count: int = 12) -> pd.DataFrame:
    rows = []

    for index in range(row_count):
        close = 100.0 + index
        rows.append(
            {
                "event_ts": pd.Timestamp("2026-01-01", tz="UTC")
                + pd.Timedelta(days=index),
                "symbol": "AAPL",
                "open": close - 1,
                "high": close + 2,
                "low": close - 2,
                "close": close,
                "volume": 1_000_000 + index * 10_000,
                "source": "unit_test",
                "snapshot_id": "snapshot_001",
                "ingested_at": pd.Timestamp("2026-01-01T23:00:00Z")
                + pd.Timedelta(days=index),
            }
        )

    return pd.DataFrame(rows)

def test_feature_snapshot_passes_contract() -> None:
    # TODO: load fixture
    market_bars  = make_market_bars()
    # TODO: build feature snapshot
    features_snapshot = build_feature_snapshot(market_bars,feature_set_version="pit_features_v1",generated_at=GENERATED_AT,)
    # TODO: validate expected columns / contract
    validate_feature_snapshot(features_snapshot)


def test_feature_snapshot_is_deterministic() -> None:
    # TODO: same input + same generated_at should produce identical output
    market_bars = make_market_bars()

    first = build_feature_snapshot(
        market_bars,
        feature_set_version="pit_features_v1",
        generated_at=GENERATED_AT,
    )
    second = build_feature_snapshot(
        market_bars,
        feature_set_version="pit_features_v1",
        generated_at=GENERATED_AT,
    )

    pd.testing.assert_frame_equal(first, second)


def test_feature_builder_does_not_mutate_input() -> None:
    # TODO: deep-copy input, run builder, compare original
    market_bars = make_market_bars()
    original = market_bars.copy(deep=True)

    build_feature_snapshot(
        market_bars,
        feature_set_version="pit_features_v1",
        generated_at=GENERATED_AT,
    )

    pd.testing.assert_frame_equal(market_bars, original)

def test_adding_future_row_does_not_change_existing_features() -> None:
    # TODO:
    # - build features from original fixture
    # - append a later event_ts row
    # - rebuild features
    # - assert all original feature rows are unchanged
    market_bars = make_market_bars()
    original_features = build_feature_snapshot(
        market_bars,
        generated_at=GENERATED_AT,
    )
    future_row = market_bars.iloc[-1].copy()
    future_row["event_ts"] = future_row["event_ts"] + pd.Timedelta(days=1)
    future_row["open"] = 120.0
    future_row["high"] = 123.0
    future_row["low"] = 119.0
    future_row["close"] = 122.0
    future_row["volume"] = 2_000_000
    future_row["ingested_at"] = future_row["ingested_at"] + pd.Timedelta(days=1)

    expanded_bars = pd.concat(
        [market_bars, pd.DataFrame([future_row])],
        ignore_index=True,
    )

    expanded_features = build_feature_snapshot(
        expanded_bars,
        generated_at=GENERATED_AT,
    )

    comparable_expanded = expanded_features[
        expanded_features["event_ts"].isin(original_features["event_ts"])
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(original_features, comparable_expanded)



def test_changing_future_row_does_not_change_past_features() -> None:
    # TODO:
    # - build features from fixture
    # - modify only a future row
    # - rebuild features
    # - assert rows before that future date are unchanged
    market_bars = make_market_bars()
    original_features = build_feature_snapshot(
        market_bars,
        generated_at=GENERATED_AT,
    )

    changed_bars = market_bars.copy()
    changed_bars.loc[changed_bars.index[-1], "close"] = 150.0
    changed_bars.loc[changed_bars.index[-1], "high"] = 155.0

    changed_features = build_feature_snapshot(
        changed_bars,
        generated_at=GENERATED_AT,
    )

    cutoff_ts = market_bars.iloc[-1]["event_ts"]

    original_past = original_features[
        original_features["event_ts"] < cutoff_ts
    ].reset_index(drop=True)
    changed_past = changed_features[
        changed_features["event_ts"] < cutoff_ts
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(original_past, changed_past)


def test_insufficient_history_rows_are_dropped_or_handled() -> None:
    # TODO:
    # - use a tiny fixture with too few rows for rolling windows
    # - assert expected empty/smaller output
    market_bars = make_market_bars(row_count=5)

    feature_snapshot = build_feature_snapshot(
        market_bars,
        generated_at=GENERATED_AT,
    )

    assert feature_snapshot.empty


def test_invalid_market_bars_fail_before_feature_generation() -> None:
    # TODO:
    # - corrupt input data
    # - assert builder raises ValueError
    market_bars = make_market_bars()
    market_bars.loc[0, "close"] = -1

    with pytest.raises(ValueError, match="Price column must be positive"):
        build_feature_snapshot(
            market_bars,
            generated_at=GENERATED_AT,
        )