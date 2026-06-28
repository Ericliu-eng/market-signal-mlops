from __future__ import annotations

from pathlib import Path

import pandas as pd

from market_signal_mlops.validation.feature_snapshots import validate_feature_snapshot
from market_signal_mlops.validation.market_bars import validate_market_bars


DEFAULT_FEATURE_SET_VERSION = "basic_v1"


def build_basic_feature_snapshot(
    market_bars: pd.DataFrame,
    *,
    feature_set_version: str = DEFAULT_FEATURE_SET_VERSION,
    generated_at: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """
    Build deterministic point-in-time features from validated market bars.

    Features use only same-row values or prior rows within the same symbol.
    Rows without enough history for lagged features are dropped.
    """
    validate_market_bars(market_bars)

    if generated_at is None:
        generated_at = pd.Timestamp.now(tz="UTC")
    else:
        generated_at = pd.Timestamp(generated_at)

    bars = market_bars.sort_values(["symbol", "event_ts"]).reset_index(drop=True)
    features = bars[
        [
            "event_ts",
            "symbol",
            "snapshot_id",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ].copy()

    grouped = features.groupby("symbol", sort=False)

    features["feature_set_version"] = feature_set_version
    features["generated_at"] = generated_at
    features["return_1d"] = grouped["close"].pct_change()
    features["volume_change_1d"] = grouped["volume"].pct_change()
    features["intraday_return"] = (features["close"] / features["open"]) - 1
    features["high_low_range"] = (
        (features["high"] - features["low"]) / features["close"]
    )

    output_columns = [
        "event_ts",
        "symbol",
        "snapshot_id",
        "feature_set_version",
        "generated_at",
        "return_1d",
        "volume_change_1d",
        "intraday_return",
        "high_low_range",
    ]
    feature_snapshot = features[output_columns].dropna().reset_index(drop=True)

    validate_feature_snapshot(feature_snapshot)
    return feature_snapshot


def main() -> None:
    fixture_path = Path("data/fixtures/market_bars.csv")
    market_bars = pd.read_csv(
        fixture_path,
        parse_dates=["event_ts", "ingested_at"],
    )
    feature_snapshot = build_basic_feature_snapshot(
        market_bars,
        feature_set_version=DEFAULT_FEATURE_SET_VERSION,
        generated_at=pd.Timestamp.now(tz="UTC"),
    )

    print(feature_snapshot.to_string(index=False))


if __name__ == "__main__":
    main()
