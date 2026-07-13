# src/market_signal_mlops/features/labels.py

from __future__ import annotations

import pandas as pd

from market_signal_mlops.validation.market_bars import validate_market_bars


def build_next_day_volatility_labels(
    market_bars: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build primary labels for next-day volatility prediction.

    Label idea:
    - For each symbol/date t, label uses information from t+1.
    - Features for t must never include this future value.

    TODO:
    - Validate input market bars.
    - Sort by symbol/event_ts.
    - Compute next-day return per symbol.
    - Compute absolute next-day return as volatility target.
    - Return event_ts, symbol, snapshot_id, label column.
    - Drop rows where next-day label is unavailable.
    """

    # TODO: validate_market_bars(market_bars)

    validate_market_bars(market_bars)

    # TODO: sort by symbol/event_ts
    bars = (
        market_bars
        .copy()
        .sort_values(
            ["symbol", "event_ts"],
            ascending=[True, True],
        )
        .reset_index(drop=True)
    )

    # TODO: compute next_day_return
        # 3. Get the next day's closing price within each symbol
    bars["next_day_close"] = (
        bars
        .groupby("symbol")["close"]
        .shift(-1)
    )
    # TODO: compute target_next_day_abs_return
    bars["next_day_return"] = (
        bars["next_day_close"] / bars["close"] - 1
    )
    bars["target_next_day_abs_return"] = (
        bars["next_day_return"].abs()
    )    
    
    # TODO: select output columns
    labels = bars[
        [
            "event_ts",
            "symbol",
            "snapshot_id",
            "target_next_day_abs_return",
        ]
    ].copy()
    # 7. Last date for each symbol has no next-day value
    labels = labels.dropna(
        subset=["target_next_day_abs_return"]
    )

    return labels.reset_index(drop=True)


def build_next_day_direction_labels(
    market_bars: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build optional secondary labels for next-day direction prediction.

    TODO:
    - Compute whether next-day return is positive.
    - Keep this separate from feature generation.
    """
    labels = market_bars.sort_values(["symbol", "event_ts"]).reset_index(drop=True)
    labels = labels[["event_ts", "symbol", "snapshot_id", "close"]].copy()

    grouped = labels.groupby("symbol", sort=False)

    labels["next_close"] = grouped["close"].shift(-1)
    labels["next_day_return"] = (labels["next_close"] / labels["close"]) - 1

    labels = labels.dropna(subset=["next_day_return"]).reset_index(drop=True)
    labels["target_next_day_direction"] = (
        labels["next_day_return"] > 0
    ).astype(int)

    output_columns = [
        "event_ts",
        "symbol",
        "snapshot_id",
        "target_next_day_direction",
    ]
    return labels[output_columns].reset_index(drop=True)