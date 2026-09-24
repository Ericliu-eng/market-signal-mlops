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

from market_signal_mlops.registry.policy import (
    CandidateEvidence,
    PromotionPolicy,
)
from market_signal_mlops.registry.service import RegistryService

REGISTERED_MODEL_NAME = "market-signal-classifier"
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

    with mlflow.start_run(run_name="walk-forward-evaluation") as run:
        mlflow.set_tags(
            {
                "run_type": "walk_forward_evaluation",
                "challenger_model": CHALLENGER_MODEL_NAME,
                "snapshot_id": str(features["snapshot_id"].iloc[0]),
                "feature_set_version": str(features["feature_set_version"].iloc[0]),
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

        model_info = mlflow.sklearn.log_model(
            sk_model=challenger_model,
            name="challenger_model",
            input_example=input_example,
            signature=signature,
            skops_trusted_types=["numpy.dtype"],
        )

        challenger_metrics = result.aggregate_metrics.loc[
            result.aggregate_metrics["model_name"] == CHALLENGER_MODEL_NAME
        ].iloc[0]

        evidence = CandidateEvidence(
            balanced_accuracy_mean=float(challenger_metrics["balanced_accuracy_mean"]),
            balanced_accuracy_std=float(challenger_metrics["balanced_accuracy_std"]),
            roc_auc_mean=float(challenger_metrics["roc_auc_mean"]),
            brier_score_mean=float(challenger_metrics["brier_score_mean"]),
            data_contract_passed=True,
            signature_present=signature is not None,
        )

        decision = PromotionPolicy().evaluate(evidence)

        registry = RegistryService(REGISTERED_MODEL_NAME)
        candidate = registry.register_candidate(
            model_uri=model_info.model_uri,
            run_id=run.info.run_id,
        )

        if decision.approved:
            registry.promote_candidate(
                version=candidate.version,
                decision=decision,
            )

        mlflow.set_tags(
            {
                "promotion_gate_approved": str(decision.approved).lower(),
                "registered_model_name": candidate.name,
                "registered_model_version": candidate.version,
            }
        )

        mlflow.log_dict(
            {
                "approved": decision.approved,
                "reasons": list(decision.reasons),
                "registered_model_name": candidate.name,
                "registered_model_version": candidate.version,
            },
            "promotion_decision.json",
        )


if __name__ == "__main__":
    main()
