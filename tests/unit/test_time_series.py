from __future__ import annotations

import pandas as pd
import pytest

from market_signal_mlops.evaluation.time_series import ExpandingWindowSplitter


def test_expanding_window_keeps_training_before_validation() -> None:
    timestamps = pd.date_range("2026-01-01", periods=12, freq="D")

    splitter = ExpandingWindowSplitter(
        min_train_size=4,
        validation_size=2,
        n_splits=3,
    )

    folds = list(splitter.split(timestamps))

    assert len(folds) == 3

    for train_indices, validation_indices in folds:
        assert len(set(train_indices).intersection(validation_indices)) == 0
        assert max(train_indices) < min(validation_indices)
        assert timestamps[max(train_indices)] < timestamps[min(validation_indices)]



def test_training_window_expands_for_each_fold() -> None:
    timestamps = pd.date_range("2026-01-01", periods=12, freq="D")

    splitter = ExpandingWindowSplitter(
        min_train_size=4,
        validation_size=2,
        n_splits=3,
    )

    folds = list(splitter.split(timestamps))

    assert [len(train) for train, _ in folds] == [4, 6, 8]
    assert [len(validation) for _, validation in folds] == [2, 2, 2]


def test_splitter_rejects_insufficient_timestamps() -> None:
    timestamps = pd.date_range("2026-01-01", periods=5, freq="D")

    splitter = ExpandingWindowSplitter(
        min_train_size=4,
        validation_size=2,
        n_splits=2,
    )

    with pytest.raises(ValueError, match="Not enough timestamps"):
        list(splitter.split(timestamps))



def test_same_timestamp_never_appears_in_both_windows() -> None:
    timestamps = pd.to_datetime(
        [
            "2026-01-01",
            "2026-01-01",
            "2026-01-02",
            "2026-01-02",
            "2026-01-03",
            "2026-01-03",
            "2026-01-04",
            "2026-01-04",
        ]
    )

    splitter = ExpandingWindowSplitter(
        min_train_size=2,
        validation_size=1,
        n_splits=2,
    )

    for train_indices, validation_indices in splitter.split(timestamps):
        train_dates = set(timestamps[train_indices])
        validation_dates = set(timestamps[validation_indices])

        assert train_dates.isdisjoint(validation_dates)