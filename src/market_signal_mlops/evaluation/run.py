from pathlib import Path

import pandas as pd
import mlflow
import subprocess


from market_signal_mlops.evaluation.evaluator import (
    EvaluationConfig,
    TimeSeriesEvaluator,
)
from market_signal_mlops.evaluation.reporting import write_evaluation_report
from market_signal_mlops.features.feature_builder import build_feature_snapshot
from market_signal_mlops.features.labels import build_next_day_direction_labels


def main() -> None:

    
    mlflow.set_experiment("market-signal-next-day-direction-v2")

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

    with mlflow.start_run(run_name="walk-forward-evaluation"):
        mlflow.set_tags(
    {
        "run_type": "walk_forward_evaluation",
        "snapshot_id": str(features["snapshot_id"].iloc[0]),
        "feature_set_version": str(
            features["feature_set_version"].iloc[0]),
        "git_sha": subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip(),    

        })
        mlflow.log_params(
            {
                "target_column": evaluator.config.target_column,
                "min_train_size": evaluator.config.min_train_size,
                "validation_size": evaluator.config.validation_size,
                "n_splits": evaluator.config.n_splits,
            }
        )        

        result = evaluator.evaluate(features, labels)

        write_evaluation_report(
            result,
            Path("artifacts/evaluations/run-001"),
        )
        for record in result.aggregate_metrics.to_dict(orient="records"):
            model_name = record.pop("model_name")
            mlflow.log_metrics(
                {
                    f"{model_name}_{metric_name}": float(metric_value)
                    for metric_name, metric_value in record.items()
                }
            )        
        mlflow.log_artifacts("artifacts/evaluations/run-001")

if __name__ == "__main__":
    main()