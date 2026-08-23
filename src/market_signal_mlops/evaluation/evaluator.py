from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from market_signal_mlops.evaluation.time_series import ExpandingWindowSplitter


IDENTIFIER_COLUMNS = {
    "event_ts", "symbol", "snapshot_id", "feature_set_version", "generated_at"
}
JOIN_COLUMNS = ["event_ts", "symbol", "snapshot_id"]


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for one deterministic walk-forward evaluation."""

    target_column: str
    min_train_size: int
    validation_size: int
    n_splits: int
    positive_label: int = 1


@dataclass
class EvaluationResult:
    """All outputs required to inspect and reproduce an evaluation."""

    fold_metrics: pd.DataFrame
    aggregate_metrics: pd.DataFrame
    predictions: pd.DataFrame
    fold_boundaries: pd.DataFrame


class TimeSeriesEvaluator:
    """Evaluate leakage-safe classification baselines over time-ordered folds."""

    def __init__(self, config: EvaluationConfig) -> None:
        if not config.target_column.strip():
            raise ValueError("target_column must be non-empty")
        if config.min_train_size < 1:
            raise ValueError("min_train_size must be positive")
        if config.validation_size < 1:
            raise ValueError("validation_size must be positive")
        if config.n_splits < 1:
            raise ValueError("n_splits must be positive")

        self.config = config
        self.splitter = ExpandingWindowSplitter(
            min_train_size=config.min_train_size,
            validation_size=config.validation_size,
            n_splits=config.n_splits,
        )

    def evaluate(
        self, feature_snapshot: pd.DataFrame, labels: pd.DataFrame
    ) -> EvaluationResult:
        frame = self._prepare_modeling_frame(feature_snapshot, labels)
        feature_columns = self._feature_columns(frame)

        fold_metric_frames: list[pd.DataFrame] = []
        prediction_frames: list[pd.DataFrame] = []
        boundary_frames: list[pd.DataFrame] = []
        for fold_number, (train_indices, validation_indices) in enumerate(
            self.splitter.split(frame["event_ts"]), start=1
        ):
            fold_metrics, predictions, boundaries = self._evaluate_fold(
                fold_number=fold_number,
                frame=frame,
                feature_columns=feature_columns,
                train_indices=train_indices,
                validation_indices=validation_indices,
            )
            fold_metric_frames.append(fold_metrics)
            prediction_frames.append(predictions)
            boundary_frames.append(boundaries)

        fold_metrics = pd.concat(fold_metric_frames, ignore_index=True)
        return EvaluationResult(
            fold_metrics=fold_metrics,
            aggregate_metrics=self._aggregate_metrics(fold_metrics),
            predictions=pd.concat(prediction_frames, ignore_index=True),
            fold_boundaries=pd.concat(boundary_frames, ignore_index=True),
        )

    def _prepare_modeling_frame(
        self, feature_snapshot: pd.DataFrame, labels: pd.DataFrame
    ) -> pd.DataFrame:
        for name, dataframe in (("feature_snapshot", feature_snapshot), ("labels", labels)):
            missing = [column for column in JOIN_COLUMNS if column not in dataframe]
            if missing:
                raise ValueError(f"{name} missing required columns: {missing}")
        if self.config.target_column not in labels:
            raise ValueError(f"labels missing target column: {self.config.target_column}")

        frame = feature_snapshot.merge(
            labels[JOIN_COLUMNS + [self.config.target_column]],
            on=JOIN_COLUMNS,
            how="inner",
            validate="one_to_one",
        )
        if frame.empty:
            raise ValueError("No rows remain after joining features and labels")
        if frame[self.config.target_column].isna().any():
            raise ValueError(f"Target column contains missing values: {self.config.target_column}")
        return frame.sort_values(["event_ts", "symbol"], kind="stable").reset_index(
            drop=True
        )

    def _feature_columns(self, frame: pd.DataFrame) -> list[str]:
        excluded_columns = IDENTIFIER_COLUMNS | {self.config.target_column}
        feature_columns = sorted(
            column
            for column in frame.columns
            if column not in excluded_columns
            and pd.api.types.is_numeric_dtype(frame[column])
        )
        if not feature_columns:
            raise ValueError("No numeric feature columns available for modeling")
        if not np.isfinite(frame[feature_columns].to_numpy(dtype=float)).all():
            raise ValueError("Feature columns contain null or non-finite values")
        return feature_columns

    def _candidate_models(self) -> dict[str, Any]:
        return {
            "naive_prior": DummyClassifier(strategy="prior"),
            "logistic_regression": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=1_000, random_state=42)),
                ]
            ),
        }

    def _evaluate_fold(
        self,
        *,
        fold_number: int,
        frame: pd.DataFrame,
        feature_columns: list[str],
        train_indices: list[int],
        validation_indices: list[int],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_frame = frame.iloc[train_indices]
        validation_frame = frame.iloc[validation_indices]
        train_end = train_frame["event_ts"].max()
        validation_start = validation_frame["event_ts"].min()
        if train_end >= validation_start:
            raise ValueError(f"Time leakage detected in fold {fold_number}")

        x_train = train_frame[feature_columns]
        y_train = train_frame[self.config.target_column]
        x_validation = validation_frame[feature_columns]
        y_validation = validation_frame[self.config.target_column]
        if y_train.nunique() < 2:
            raise ValueError(f"Fold {fold_number} training target has only one class")

        metric_rows: list[dict[str, Any]] = []
        prediction_frames: list[pd.DataFrame] = []
        for model_name, model in self._candidate_models().items():
            model.fit(x_train, y_train)
            predictions = pd.Series(model.predict(x_validation), index=validation_frame.index)
            probabilities = model.predict_proba(x_validation)
            positive_index = list(model.classes_).index(self.config.positive_label)
            positive_probability = pd.Series(
                probabilities[:, positive_index], index=validation_frame.index
            )
            metric_rows.append(
                {
                    "fold_number": fold_number,
                    "model_name": model_name,
                    **self._classification_metrics(
                        y_true=y_validation,
                        y_pred=predictions,
                        y_probability=positive_probability,
                    ),
                }
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "fold_number": fold_number,
                        "model_name": model_name,
                        "event_ts": validation_frame["event_ts"].to_numpy(),
                        "symbol": validation_frame["symbol"].to_numpy(),
                        "snapshot_id": validation_frame["snapshot_id"].to_numpy(),
                        "y_true": y_validation.to_numpy(),
                        "y_pred": predictions.to_numpy(),
                        "y_probability": positive_probability.to_numpy(),
                    }
                )
            )

        boundaries = pd.DataFrame(
            [
                {
                    "fold_number": fold_number,
                    "train_start": train_frame["event_ts"].min(),
                    "train_end": train_end,
                    "validation_start": validation_start,
                    "validation_end": validation_frame["event_ts"].max(),
                    "train_row_count": len(train_frame),
                    "validation_row_count": len(validation_frame),
                }
            ]
        )
        return pd.DataFrame(metric_rows), pd.concat(prediction_frames), boundaries

    def _classification_metrics(
        self,
        *,
        y_true: pd.Series,
        y_pred: pd.Series,
        y_probability: pd.Series,
    ) -> dict[str, float]:
        y_true_binary = (y_true == self.config.positive_label).astype(int)
        roc_auc = np.nan
        if y_true_binary.nunique() == 2:
            roc_auc = roc_auc_score(y_true_binary, y_probability)
        return {
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    pos_label=self.config.positive_label,
                    zero_division=0,
                )
            ),
            "roc_auc": float(roc_auc),
            "brier_score": float(brier_score_loss(y_true_binary, y_probability)),
        }

    def _aggregate_metrics(self, fold_metrics: pd.DataFrame) -> pd.DataFrame:
        metric_columns = ["balanced_accuracy", "f1", "roc_auc", "brier_score"]
        aggregate = fold_metrics.groupby("model_name")[metric_columns].agg(["mean", "std"])
        aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
        aggregate["fold_count"] = fold_metrics.groupby("model_name").size()
        return aggregate.reset_index()
