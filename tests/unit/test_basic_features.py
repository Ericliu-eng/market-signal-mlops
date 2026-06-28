from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from market_signal_mlops.features import build_basic_feature_snapshot
from market_signal_mlops.validation.feature_snapshots import validate_feature_snapshot


FIXTURE_PATH = Path("data/fixtures/market_bars.csv")
GENERATED_AT = pd.Timestamp("2026-01-06T00:00:00Z")


def load_market_bars() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_PATH, parse_dates=["event_ts", "ingested_at"])


def test_basic_feature_snapshot_passes_contract() -> None:
    market_bars = load_market_bars()

    feature_snapshot = build_basic_feature_snapshot(
        market_bars,
        feature_set_version="basic_v1",
        generated_at=GENERATED_AT,
    )

    validate_feature_snapshot(feature_snapshot)


def test_basic_features_are_deterministic_for_fixed_inputs() -> None:
    market_bars = load_market_bars()

    first = build_basic_feature_snapshot(
        market_bars,
        feature_set_version="basic_v1",
        generated_at=GENERATED_AT,
    )
    second = build_basic_feature_snapshot(
        market_bars,
        feature_set_version="basic_v1",
        generated_at=GENERATED_AT,
    )

    pd.testing.assert_frame_equal(first, second)


def test_basic_features_do_not_mutate_input() -> None:
    market_bars = load_market_bars()
    original = market_bars.copy(deep=True)

    build_basic_feature_snapshot(
        market_bars,
        feature_set_version="basic_v1",
        generated_at=GENERATED_AT,
    )

    pd.testing.assert_frame_equal(market_bars, original)


def test_future_row_change_does_not_change_past_feature_row() -> None:
    market_bars = load_market_bars()
    original_features = build_basic_feature_snapshot(
        market_bars,
        feature_set_version="basic_v1",
        generated_at=GENERATED_AT,
    )

    changed_future = market_bars.copy()
    future_row_index = 2
    changed_future.loc[future_row_index, "close"] = (
        changed_future.loc[future_row_index, "low"]
        + changed_future.loc[future_row_index, "high"]
    ) / 2
    changed_features = build_basic_feature_snapshot(
        changed_future,
        feature_set_version="basic_v1",
        generated_at=GENERATED_AT,
    )

    original_past_row = original_features[
        (original_features["symbol"] == "AAPL")
        & (original_features["event_ts"] == pd.Timestamp("2026-01-02"))
    ].reset_index(drop=True)
    changed_past_row = changed_features[
        (changed_features["symbol"] == "AAPL")
        & (changed_features["event_ts"] == pd.Timestamp("2026-01-02"))
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(original_past_row, changed_past_row)


def test_invalid_market_bars_fail_before_feature_generation() -> None:
    market_bars = load_market_bars()
    market_bars.loc[0, "close"] = -1

    with pytest.raises(ValueError, match="Price column must be positive"):
        build_basic_feature_snapshot(
            market_bars,
            feature_set_version="basic_v1",
            generated_at=GENERATED_AT,
        )
