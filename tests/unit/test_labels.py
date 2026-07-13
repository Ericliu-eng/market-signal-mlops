
import pandas as pd
import pytest
from market_signal_mlops.features.labels import (
    build_next_day_direction_labels,
    build_next_day_volatility_labels,
)

def make_market_bars() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_ts": pd.Timestamp("2026-01-01", tz="UTC"),
                "symbol": "AAPL",
                "open": 100.0,
                "high": 106.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1_000_000,
                "source": "unit_test",
                "snapshot_id": "snapshot_001",
                "ingested_at": pd.Timestamp("2026-01-01T23:00:00Z"),
            },
            {
                "event_ts": pd.Timestamp("2026-01-02", tz="UTC"),
                "symbol": "AAPL",
                "open": 110.0,
                "high": 112.0,
                "low": 108.0,
                "close": 110.0,
                "volume": 1_100_000,
                "source": "unit_test",
                "snapshot_id": "snapshot_001",
                "ingested_at": pd.Timestamp("2026-01-02T23:00:00Z"),
            },
            {
                "event_ts": pd.Timestamp("2026-01-03", tz="UTC"),
                "symbol": "AAPL",
                "open": 105.0,
                "high": 107.0,
                "low": 103.0,
                "close": 105.0,
                "volume": 1_200_000,
                "source": "unit_test",
                "snapshot_id": "snapshot_001",
                "ingested_at": pd.Timestamp("2026-01-03T23:00:00Z"),
            },
        ]
    )

def test_next_day_volatility_labels_are_shifted_forward() -> None:
    market_bars = make_market_bars()

    labels = build_next_day_volatility_labels(market_bars)
    #next_return = (next_day  close / today_close) - 1 
    first_label = labels.loc[
        labels["event_ts"] == pd.Timestamp("2026-01-01", tz="UTC"),
        "target_next_day_abs_return",
    ].iloc[0]

    assert first_label == pytest.approx(0.10)

def test_last_row_per_symbol_has_no_next_day_label() -> None:
    market_bars = make_market_bars()

    labels = build_next_day_volatility_labels(market_bars)
    assert pd.Timestamp("2026-01-03", tz="UTC") not in set(labels["event_ts"])

def test_direction_labels_are_binary() -> None :
    bar = make_market_bars()
    direct_label = build_next_day_direction_labels(bar)
    
    assert direct_label["target_next_day_direction"].isin([0, 1]).all()



def test_label_builder_does_not_mutate_input() -> None:
    market_bars = make_market_bars()
    original = market_bars.copy(deep=True)

    build_next_day_volatility_labels(market_bars)
    build_next_day_direction_labels(market_bars)

    pd.testing.assert_frame_equal(market_bars, original)