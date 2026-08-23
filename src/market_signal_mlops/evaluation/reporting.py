from __future__ import annotations

from pathlib import Path

from market_signal_mlops.evaluation.evaluator import EvaluationResult


def write_evaluation_report(
    result: EvaluationResult,
    output_dir: Path,
) -> None:
    """Persist a reproducible walk-forward evaluation report."""

    output_dir.mkdir(parents=True, exist_ok=True)

    result.fold_metrics.to_csv(
        output_dir / "fold_metrics.csv",
        index=False,
    )
    result.aggregate_metrics.to_csv(
        output_dir / "aggregate_metrics.csv",
        index=False,
    )
    result.predictions.to_csv(
        output_dir / "predictions.csv",
        index=False,
    )
    result.fold_boundaries.to_csv(
        output_dir / "fold_boundaries.csv",
        index=False,
    )

