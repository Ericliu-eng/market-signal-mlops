from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from market_signal_mlops.validation.market_bars import validate_market_bars


FIXTURE_PATH = Path("data/fixtures/market_bars_sample.csv")


def load_fixture() -> pd.DataFrame:
    
    return pd.read_csv(FIXTURE_PATH, parse_dates=["event_ts", "ingested_at"])


def test_valid_fixture_passes_contract() -> None:
    df = load_fixture()
    validate_market_bars(df)


def test_missing_required_column_fails() -> None:
    df = load_fixture()
    df = df.drop(columns=["close"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_market_bars(df)


def test_duplicate_primary_key_fails() -> None:
    df = load_fixture()
    duplicated = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate primary key"):
        validate_market_bars(duplicated)


def test_negative_price_fails() -> None:
    df = load_fixture()
    df.loc[0, "close"] = -1

    with pytest.raises(ValueError, match="Price column must be positive"):
        validate_market_bars(df)


def test_negative_volume_fails() -> None:
    df = load_fixture()
    df.loc[0, "volume"] = -100

    with pytest.raises(ValueError, match="Volume must be non-negative"):
        validate_market_bars(df)


def test_null_required_value_fails() -> None:
    df = load_fixture()
    df.loc[0, "symbol"] = None

    with pytest.raises(ValueError, match="Required columns contain null values"):
        validate_market_bars(df)