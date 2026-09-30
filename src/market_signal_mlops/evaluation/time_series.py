from __future__ import annotations

from collections.abc import Iterator, Sequence

import pandas as pd


class ExpandingWindowSplitter:
    """Create expanding time-ordered folds using distinct timestamps."""

    def __init__(
        self,
        min_train_size: int,
        validation_size: int,
        n_splits: int,
    ) -> None:
        if min_train_size < 1:
            raise ValueError("min_train_size must be at least 1")
        if validation_size < 1:
            raise ValueError("validation_size must be at least 1")
        if n_splits < 1:
            raise ValueError("n_splits must be at least 1")

        self.min_train_size = min_train_size
        self.validation_size = validation_size
        self.n_splits = n_splits

    def split(
        self,
        timestamps: Sequence[object],
    ) -> Iterator[tuple[list[int], list[int]]]:
        timestamp_index = pd.Index(timestamps)
        unique_timestamps = timestamp_index.unique()

        if not unique_timestamps.is_monotonic_increasing:
            raise ValueError("timestamps must be sorted in chronological order")

        required_timestamps = self.min_train_size + (
            self.validation_size * self.n_splits
        )
        if len(unique_timestamps) < required_timestamps:
            raise ValueError(
                "Not enough timestamps for the requested training and validation folds"
            )

        for fold_number in range(self.n_splits):
            train_end = self.min_train_size + (fold_number * self.validation_size)
            validation_end = train_end + self.validation_size

            train_timestamps = unique_timestamps[:train_end]
            validation_timestamps = unique_timestamps[train_end:validation_end]

            train_indices = [
                index
                for index, timestamp in enumerate(timestamp_index)
                if timestamp in train_timestamps
            ]
            validation_indices = [
                index
                for index, timestamp in enumerate(timestamp_index)
                if timestamp in validation_timestamps
            ]

            yield train_indices, validation_indices
