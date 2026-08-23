from pathlib import Path

import pandas as pd

from market_signal_mlops.evaluation.evaluator import (
    EvaluationConfig,
    TimeSeriesEvaluator,
)
from market_signal_mlops.evaluation.reporting import write_evaluation_report
from market_signal_mlops.features.feature_builder import build_feature_snapshot
from market_signal_mlops.features.labels import build_next_day_direction_labels


def main() -> None:
    bars = pd.read_csv(
        "data/fixtures/market_bars.csv",
        parse_dates=["event_ts", "ingested_at"],
    )

    features = build_feature_snapshot(bars)
    labels = build_next_day_direction_labels(bars)

    evaluator = TimeSeriesEvaluator(
        EvaluationConfig(
            target_column="target_next_day_direction",
            min_train_size=3,
            validation_size=2,
            n_splits=3,
        )
    )

    result = evaluator.evaluate(features, labels)

    write_evaluation_report(
        result,
        Path("artifacts/evaluations/run-001"),
    )


if __name__ == "__main__":
    main()