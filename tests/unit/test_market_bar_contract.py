from __future__ import annotations

from pathlib import Path

import numpy as np
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


def test_out_of_order_event_ts_fails() -> None:
    df = load_fixture()

    out_of_order = pd.concat(
        [df.iloc[[1]], df.iloc[[0]], df.iloc[2:]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="rows must be ordered by symbol and event_ts"):
        validate_market_bars(out_of_order)


@pytest.mark.parametrize("column", ["event_ts", "ingested_at"])
def test_datetime_column_requires_datetime_dtype(column: str) -> None:
    df = load_fixture()
    df[column] = "123"

    with pytest.raises(ValueError, match=f"{column} must be datetime"):
        validate_market_bars(df)


@pytest.mark.parametrize("column", ["symbol", "source", "snapshot_id"])
def test_string_column_requires_string_values(column: str) -> None:
    df = load_fixture()
    df[column] = 123

    with pytest.raises(ValueError, match=f"{column} must be string"):
        validate_market_bars(df)


@pytest.mark.parametrize("column", ["symbol", "source", "snapshot_id"])
def test_string_column_rejects_blank_values(column: str) -> None:
    df = load_fixture()
    df[column] = ""

    with pytest.raises(ValueError, match=f"{column} must be not null"):
        validate_market_bars(df)


@pytest.mark.parametrize("column", ["open", "high", "low", "close"])
def test_price_column_requires_numeric_dtype(column: str) -> None:
    df = load_fixture()
    df[column] = "hello word"

    with pytest.raises(ValueError, match=f"{column} must be number"):
        validate_market_bars(df)


@pytest.mark.parametrize("column", ["open", "high", "low", "close"])
@pytest.mark.parametrize("bad_value", [np.inf, -np.inf])
def test_price_column_rejects_non_finite_values(column: str, bad_value: float) -> None:
    df = load_fixture()

    df[column] = df[column].astype(float)
    df.loc[0, column] = bad_value

    with pytest.raises(ValueError, match=f"{column} must contain finite values"):
        validate_market_bars(df)


def test_volume_requires_integer_dtype() -> None:
    df = load_fixture()
    df["volume"] = df["volume"].astype(float)

    with pytest.raises(ValueError, match="volume must be integer"):
        validate_market_bars(df)


def test_rows_must_be_ordered_by_symbol_and_event_ts() -> None:
    df = load_fixture()

    out_of_order = pd.concat(
        [df.iloc[[1]], df.iloc[[0]], df.iloc[2:]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="rows must be ordered by symbol and event_ts"):
        validate_market_bars(out_of_order)


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        ("high", 50),
        ("low", 200),
    ],
)
def test_invalid_ohlc_relationship_fails(
    column: str,
    invalid_value: float,
) -> None:
    df = load_fixture()
    df.loc[0, column] = invalid_value

    with pytest.raises(ValueError, match="invalid OHLC relationship"):
        validate_market_bars(df)
