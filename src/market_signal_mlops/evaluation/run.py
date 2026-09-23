import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

from market_signal_mlops.evaluation.evaluator import (
    EvaluationConfig,
    TimeSeriesEvaluator,
)
from market_signal_mlops.evaluation.reporting import write_evaluation_report
from market_signal_mlops.features.feature_builder import build_feature_snapshot
from market_signal_mlops.features.labels import build_next_day_direction_labels


EXPERIMENT_NAME = "market-signal-next-day-direction-v2"
CHALLENGER_MODEL_NAME = "hist_gradient_boosting"
OUTPUT_DIR = Path("artifacts/evaluations/run-001")


def main() -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)

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

    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()

    with mlflow.start_run(run_name="walk-forward-evaluation"):
        mlflow.set_tags(
            {
                "run_type": "walk_forward_evaluation",
                "challenger_model": CHALLENGER_MODEL_NAME,
                "snapshot_id": str(features["snapshot_id"].iloc[0]),
                "feature_set_version": str(
                    features["feature_set_version"].iloc[0]
                ),
                "git_sha": git_sha,
            }
        )

        mlflow.log_params(
            {
                "target_column": evaluator.config.target_column,
                "min_train_size": evaluator.config.min_train_size,
                "validation_size": evaluator.config.validation_size,
                "n_splits": evaluator.config.n_splits,
            }
        )

        result = evaluator.evaluate(features, labels)
        write_evaluation_report(result, OUTPUT_DIR)

        for record in result.aggregate_metrics.to_dict(orient="records"):
            model_name = record.pop("model_name")
            mlflow.log_metrics(
                {
                    f"{model_name}_{metric_name}": float(metric_value)
                    for metric_name, metric_value in record.items()
                }
            )

        mlflow.log_artifacts(str(OUTPUT_DIR))

        challenger_model, training_features = evaluator.fit_candidate_model(
            CHALLENGER_MODEL_NAME,
            features,
            labels,
        )

        input_example = training_features.head(5)
        signature = infer_signature(
            input_example,
            challenger_model.predict(input_example),
        )

        mlflow.sklearn.log_model(
            sk_model=challenger_model,
            name="challenger_model",
            input_example=input_example,
            signature=signature,
            skops_trusted_types=["numpy.dtype"],
        )


if __name__ == "__main__":
    main()