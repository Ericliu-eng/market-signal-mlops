from __future__ import annotations

import pandas as pd
from pathlib import Path
from market_signal_mlops.validation.feature_snapshots import validate_feature_snapshot
from market_signal_mlops.validation.market_bars import validate_market_bars


DEFAULT_FEATURE_SET_VERSION = "pit_features_v1"


def build_feature_snapshot(
    market_bars: pd.DataFrame,
    *,
    feature_set_version: str = DEFAULT_FEATURE_SET_VERSION,
    generated_at: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """
    Build point-in-time market features from validated market bars.
    """

    # TODO: validate_market_bars(market_bars)
    validate_market_bars(market_bars)

    # TODO: normalize generated_at
    if generated_at is None:
        generated_at = pd.Timestamp.now(tz="UTC")
    else:
        generated_at = pd.Timestamp(generated_at)
    # TODO: copy and sort input data
    bar = market_bars.sort_values(["symbol", "event_ts"]).reset_index(drop=True)
    copy = bar[
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
    
    grouped = copy.groupby("symbol")
    # TODO: compute return features
    # - return_1d
    # - return_5d
    copy["return_1d"] = grouped["close"].pct_change(1)
    copy["return_5d"] = grouped["close"].pct_change(5)

    # TODO: compute trend features
    # - close_to_ma_5d
    # - close_to_ma_10d
    copy["ma_5d"] = grouped["close"].transform(lambda x: x.rolling(window=5, min_periods=5).mean())
    copy["close_to_ma_5d"] = copy["close"] / copy["ma_5d"]

    copy["ma_10d"] = grouped["close"].transform(lambda x: x.rolling(window=10, min_periods=10).mean())
    copy["close_to_ma_10d"] = copy["close"] / copy["ma_10d"]
    # TODO: compute risk features
    # - rolling_volatility_5d
    # - high_low_range
    copy["rolling_volatility_5d"] = grouped["return_1d"].transform(lambda x: x.rolling(window=5, min_periods=5).std())
    copy["high_low_range"] = copy["high"] - copy["low"]

    # TODO: compute volume features
    # - volume_change_1d
    # - volume_zscore_5d
    copy["volume_change_1d"] = grouped["volume"].pct_change(1)
    rolling_mean = grouped["volume"].transform(
        lambda x: x.rolling(window=5, min_periods=5).mean()
    )

    rolling_std = grouped["volume"].transform(
        lambda x: x.rolling(window=5, min_periods=5).std()
    )

    copy["volume_zscore_5d"] = (
        (copy["volume"] - rolling_mean) / rolling_std
    )


    copy["feature_set_version"] = feature_set_version
    copy["generated_at"] = generated_at

    # TODO: select output columns
    output_columns = [
        "event_ts",
        "symbol",
        "snapshot_id",
        "feature_set_version",
        "generated_at",
        "return_1d",
        "return_5d",
        "close_to_ma_5d",
        "close_to_ma_10d",
        "rolling_volatility_5d",
        "high_low_range",
        "volume_change_1d",
        "volume_zscore_5d",
    ]
    # TODO: drop rows with missing feature values caused by insufficient history
    feature_snapshot= copy[output_columns].dropna().reset_index(drop=True)
    # TODO: validate_feature_snapshot(feature_snapshot)
    validate_feature_snapshot(feature_snapshot)

    return  feature_snapshot


def main():
    market_bars = pd.read_csv(
        Path("data/fixtures/market_bars.csv"),
        parse_dates=["event_ts", "ingested_at"],
    )
    build_feature_snapshot(
        market_bars,
        feature_set_version=DEFAULT_FEATURE_SET_VERSION,
        generated_at=None,
    )
if __name__ == "__main__":
    main()
